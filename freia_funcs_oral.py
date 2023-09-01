'''This Code is based on the FrEIA Framework, source: https://github.com/VLL-HD/FrEIA
It is a assembly of the necessary modules/functions from FrEIA that are needed for our purposes.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from efficient_subnet import *
import numpy as np
import config as c
from utils import *

VERBOSE = False
from scipy import linalg as la

logabs = lambda x: torch.log(torch.abs(x))


class dummy_data:
    def __init__(self, *dims):
        self.dims = dims

    @property
    def shape(self):
        return self.dims

class F_fully_connected(nn.Module):
    '''Fully connected tranformation, not reversible, but used below.'''

    def __init__(self, size_in, size, channels_hidden=None, dropout=0.0):
        super(F_fully_connected, self).__init__()
        if not channels_hidden:
            channels_hidden = 2 * size
        # print(size_in)
        # print(size)
        # print(channels_hidden)
        self.d1 = nn.Dropout(p=dropout)
        self.d2 = nn.Dropout(p=dropout)
        self.d2b = nn.Dropout(p=dropout)

        self.fc1 = nn.Linear(size_in, channels_hidden)
        self.fc2 = nn.Linear(channels_hidden, channels_hidden)
        self.fc2b = nn.Linear(channels_hidden, channels_hidden)
        self.fc3 = nn.Linear(channels_hidden, size)
        self.nl1 = nn.ReLU()
        self.nl2 = nn.ReLU()
        self.nl2b = nn.ReLU()

        self.attention = Class_Attention(dim=size_in, num_heads=4)

        self.bn = nn.BatchNorm1d(size_in)

    def forward(self, x):
        # print(x.shape)

        out = self.nl1(self.d1(self.fc1(x)))
        out = self.nl2(self.d2(self.fc2(out)))
        out = self.nl2b(self.d2b(self.fc2b(out)))
        out = self.fc3(out)
        return out


class Xception_Conv(nn.Module):
    def __init__(self, in_channel):
        super(Xception_Conv, self).__init__()
        self.depth_conv_1x1 = nn.Conv2d(in_channels=in_channel,
                                        out_channels=in_channel,
                                        kernel_size=(1, 1),
                                        stride=(1, 1),
                                        padding=0,
                                        groups=in_channel
                                        )
        self.depth_conv_3x3 = nn.Conv2d(in_channels=in_channel,
                                        out_channels=in_channel,
                                        kernel_size=(3, 3),
                                        stride=(1, 1),
                                        padding=1,
                                        groups=in_channel
                                        )
        self.depth_conv_3x3_1 = nn.Conv2d(in_channels=in_channel,
                                          out_channels=in_channel,
                                          kernel_size=(3, 3),
                                          stride=(1, 1),
                                          padding=1,
                                          groups=in_channel
                                          )
        self.depth_conv_3x3_2 = nn.Conv2d(in_channels=in_channel,
                                          out_channels=in_channel,
                                          kernel_size=(3, 3),
                                          stride=(1, 1),
                                          padding=1,
                                          groups=in_channel)

    def forward(self, input):
        x1 = self.depth_conv_1x1(input)
        x2 = self.depth_conv_3x3(input)
        x3 = self.depth_conv_3x3_2(self.depth_conv_3x3_1(input))
        x = torch.cat((x1, x2, x3), dim=1)
        return x


class F_conv(nn.Module):
    def __init__(self, in_channel, channel, channels_hidden=None,
                 kernel_size=1, subnet=None, batch_norm=False):
        super(F_conv, self).__init__()
        if not channels_hidden:
            channels_hidden = channel
        pad = kernel_size // 2
        pad_mode = 'zeros'
        self.subnet = subnet
        if self.subnet is None:
            self.conv1 = Xception_Conv(in_channel=in_channel)
            self.conv2 = nn.Conv2d(3 * in_channel, channels_hidden, kernel_size, padding=pad,
                                   bias=not batch_norm, padding_mode=pad_mode)
            self.conv3 = nn.Conv2d(channels_hidden, channel, kernel_size, padding=pad,
                                   bias=not batch_norm, padding_mode=pad_mode)
        else:
            self.conv1 = subnet(in_channel, channels_hidden, kernel_size, pad)
            self.conv2 = subnet(channels_hidden, channel, kernel_size, pad)
        self.relu = nn.LeakyReLU(0.1)
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x):
        out = self.conv3(self.relu(self.conv2(self.conv1(x))))
        out = out * self.gamma
        return out


class permute_layer(nn.Module):
    '''permutes input vector in a random but fixed way'''

    def __init__(self, dims_in, seed):
        super(permute_layer, self).__init__()
        self.in_channels = dims_in[0][0]

        np.random.seed(seed)
        # in_channels为整数，所以，随机打乱arange(x)
        self.perm = np.random.permutation(self.in_channels)
        np.random.seed()

        self.perm_inv = np.zeros_like(self.perm)
        # perm_inv是值和下标相互交换的数组
        for i, p in enumerate(self.perm):
            self.perm_inv[p] = i

        self.perm = torch.LongTensor(self.perm)
        self.perm_inv = torch.LongTensor(self.perm_inv)

    def forward(self, x, rev=False):
        # 将X按照channel进行随机打乱
        # rev = True 按照perm_inv的方式打乱否则按照perm的打乱方式
        if not rev:
            return [x[0][:, self.perm]]
        else:
            return [x[0][:, self.perm_inv]]

    def jacobian(self, x, rev=False):
        # TODO: use batch size, set as nn.Parameter so cuda() works
        return 0.

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims


class glow_coupling_layer(nn.Module):
    def __init__(self, dims_in, F_class=F_fully_connected, F_args={},
                 clamp=3.):
        super(glow_coupling_layer, self).__init__()
        channels = dims_in[0][0]
        self.ndims = len(dims_in[0])

        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2

        self.clamp = clamp
        # 没用过下面这两
        self.max_s = exp(clamp)
        self.min_s = exp(-clamp)
        # 两个网络
        self.s1 = F_class(self.split_len1, self.split_len2 * 2, **F_args)
        self.s2 = F_class(self.split_len2, self.split_len1 * 2, **F_args)

    # 将神经网络的输出激活后取指数
    def e(self, s):
        return torch.exp(self.log_e(s))

    # 激活函数
    def log_e(self, s):
        # 激活函数
        return self.clamp * 0.636 * torch.atan(s / self.clamp)

    def forward(self, x, rev=False):
        # dim,start,length,从dim上取start到start + len长度的Tensor
        x1, x2 =(x[0].narrow(1, 0, self.split_len1),
                   x[0].narrow(1, self.split_len1, self.split_len2))
     
        # 一次coupling Layer
        if not rev:
            # 将输入的channels输出两倍后，前半部分激活后取指数，后半部分直接t2
            r2 = self.s2(x2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            y1 = self.e(s2) * x1 + t2

            r1 = self.s1(y1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            y2 = self.e(s1) * x2 + t1

        else:  # names of x and y are swapped!
            r1 = self.s1(x1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            y2 = (x2 - t1) / self.e(s1)

            r2 = self.s2(y2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            y1 = (x1 - t2) / self.e(s2)
        y = torch.cat((y1, y2), 1)
        # 将输出的结果压缩到-1e6到1e6之间
        y = torch.clamp(y, -1e6, 1e6)
        # B,N,C ---- B,C,N
        return [y]

    #
    def jacobian(self, x, rev=False):
        x1, x2 = (x[0].narrow(1, 0, self.split_len1),
                   x[0].narrow(1, self.split_len1, self.split_len2))
        if not rev:
            r2 = self.s2(x2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            y1 = self.e(s2) * x1 + t2

            r1 = self.s1(y1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]

        else:  # names of x and y are swapped!
            r1 = self.s1(x1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            y2 = (x2 - t1) / self.e(s1)

            r2 = self.s2(y2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
        if isinstance(self.s1, F_conv):
            jac = (torch.sum(self.log_e(s1), dim=(1, 2, 3))
                   + torch.sum(self.log_e(s2), dim=(1, 2, 3)))
        else:
            jac = (torch.sum(self.log_e(s1), dim=1)
                   + torch.sum(self.log_e(s2), dim=1))
        for i in range(self.ndims - 1):
            jac = torch.sum(jac, dim=1)

        return jac

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims


class Node:
    '''The Node class represents one transformation in the graph, with an
    arbitrary number of in- and outputs.'''

    def __init__(self, inputs, module_type, module_args, name=None):
        # inputs是上一个节点的输出
        self.inputs = inputs
        # 当前节点的输出，去列表是因为可能输出不只有一个，在INN中输出时，可能被拆分为多份
        self.outputs = []
        self.module_type = module_type
        self.module_args = module_args

        self.input_dims, self.module = None, None
        self.computed = None
        self.computed_rev = None
        self.id = None

        if name:
            self.name = name
        else:
            # 取地址的16进制形式的后6位作名字
            self.name = hex(id(self))[-6:]
        #     给当前Node生成256个输出
        for i in range(255):
            # exec指执行指定的语句
            exec('self.out{0} = (self, {0})'.format(i))

    def build_modules(self, verbose=VERBOSE):
        ''' Returns a list with the dimension of each output of this node,
        recursively calling build_modules of the nodes connected to the input.
        Use this information to initialize the pytorch nn.Module of this node.
        '''

        if not self.input_dims:  # Only do it if this hasn't been computed yet
            # n位inputnode或者上一个node的输出对象，c为0,调用函数后返回的是datashape
            # n代表上一个的输出，c代表本身为上一个的第几个输出
            # 这里记录的是上一个节点的输出形状，这里为要输入的维度
            # 这里只有一个，一般为n_scale
            self.input_dims = [n.build_modules(verbose=verbose)[c]
                               for n, c in self.inputs]
            # 第一个一般为permute Layer
            try:
                self.module = self.module_type(self.input_dims,
                                               **self.module_args)
            except Exception as e:
                print('Error in node %s' % (self.name))
                raise e

            if verbose:
                print("Node %s has following input dimensions:" % (self.name))
                for d, (n, c) in zip(self.input_dims, self.inputs):
                    print("\t Output #%i of node %s:" % (c, n.name), d)
                print()
            # 上面每个Module返回的为输入的维度，这里也一样，列表一般只有一个，因为规定的输出只有一个
            self.output_dims = self.module.output_dims(self.input_dims)
            self.n_outputs = len(self.output_dims)

        return self.output_dims

    # 该函数绑定当前节点为上一个节点的输出，并记录了当前节点有那些需要计算的地方
    def run_forward(self, op_list):
        '''Determine the order of operations needed to reach this node. Calls
        run_forward of parent nodes recursively. Each operation is appended to
        the global list op_list, in the form (node ID, input variable IDs,
        output variable IDs)'''

        if not self.computed:

            # Compute all nodes which provide inputs, filter out the
            # channels you need
            self.input_vars = []
            # 这里是一个嵌套，从output开始，将它的输入（上一个节点的输出调用其runforward）
            # 如果上一个节点是Node就一直嵌套下去，如果是InputNode就只用返回inputNode的id和标号，再回来返回第二个（Node的标号），重复直到最后一个output
            for i, (n, c) in enumerate(self.inputs):
                self.input_vars.append(n.run_forward(op_list)[c])
                # Register youself as an output in the input node
                # 这一步相当于将自己注册为上一个节点的输出
                n.outputs.append((self, i))

            # All outputs could now be computed
            # 返回的是要进行计算的对象（包括id,和i）,这个i分别表明，有哪几个输出，INN中输出允许存在多个输出的
            self.computed = [(self.id, i) for i in range(self.n_outputs)]
            # input_var是输入节点的id,和上一个输出的信息
            op_list.append((self.id, self.input_vars, self.computed))

        # Return the variables you have computed (this happens mulitple times
        # without recomputing if called repeatedly)
        return self.computed

    def run_backward(self, op_list):
        '''See run_forward, this is the same, only for the reverse computation.
        Need to call run_forward first, otherwise this function will not
        work'''

        assert len(self.outputs) > 0, "Call run_forward first"
        if not self.computed_rev:

            # These are the input variables that must be computed first
            output_vars = [(self.id, i) for i in range(self.n_outputs)]

            # Recursively compute these
            for n, c in self.outputs:
                n.run_backward(op_list)

            # The variables that this node computes are the input variables
            # from the forward pass
            self.computed_rev = self.input_vars
            op_list.append((self.id, output_vars, self.computed_rev))

        return self.computed_rev


class InputNode(Node):
    '''Special type of node that represents the input data of the whole net (or
    ouput when running reverse)'''

    def __init__(self, *dims, name='node'):
        self.name = name
        self.data = dummy_data(*dims)
        self.outputs = []
        self.module = None
        self.computed_rev = None
        self.n_outputs = 1
        self.input_vars = []
        self.out0 = (self, 0)

    # 输入节点只用于指定输入，不用它来进行module的操作
    # 只用于返回传入数据的形状
    def build_modules(self, verbose=VERBOSE):
        return [self.data.shape]

    # 只用于返回当前对象和标号
    def run_forward(self, op_list):
        return [(self.id, 0)]


class OutputNode(Node):
    '''Special type of node that represents the output of the whole net (of the
    input when running in reverse)'''

    class dummy(nn.Module):

        def __init__(self, *args):
            super(OutputNode.dummy, self).__init__()

        def __call__(*args):
            return args

        def output_dims(*args):
            return args

    def __init__(self, inputs, name='node'):
        self.module_type, self.module_args = self.dummy, {}
        self.output_dims = []
        self.inputs = inputs
        self.input_dims, self.module = None, None
        self.computed = None
        self.id = None
        self.name = name
        # 将自己绑定为上一个的输出
        for c, inp in enumerate(self.inputs):
            inp[0].outputs.append((self, c))

    def run_backward(self, op_list):
        return [(self.id, 0)]


class ReversibleGraphNet(nn.Module):
    '''This class represents the invertible net itself. It is a subclass of
    torch.nn.Module and supports the same methods. The forward method has an
    additional option 'rev', whith which the net can be computed in reverse.'''

    def __init__(self, node_list, ind_in=None, ind_out=None, verbose=False):
        '''node_list should be a list of all nodes involved, and ind_in,
        ind_out are the indexes of the special nodes InputNode and OutputNode
        in this list.'''
        super(ReversibleGraphNet, self).__init__()

        # Gather lists of input and output nodes
        # ind_in是一个列表，记录InputNode的位置，ind_out记录OutputNode的位置
        if ind_in is not None:
            if isinstance(ind_in, int):
                self.ind_in = list([ind_in])
            else:
                self.ind_in = ind_in
        else:
            self.ind_in = [i for i in range(len(node_list))
                           if isinstance(node_list[i], InputNode)]
            assert len(self.ind_in) > 0, "No input nodes specified."
        if ind_out is not None:
            if isinstance(ind_out, int):
                self.ind_out = list([ind_out])
            else:
                self.ind_out = ind_out
        else:
            self.ind_out = [i for i in range(len(node_list))
                            if isinstance(node_list[i], OutputNode)]
            assert len(self.ind_out) > 0, "No output nodes specified."

        self.return_vars = []
        self.input_vars = []

        # Assign each node a unique ID
        self.node_list = node_list
        for i, n in enumerate(node_list):
            n.id = i

        # Recursively build the nodes nn.Modules and determine order of
        # operations
        ops = []
        for i in self.ind_out:
            node_list[i].build_modules(verbose=verbose)
            node_list[i].run_forward(ops)

        # create list of Pytorch variables that are used
        variables = set()
        for o in ops:
            variables = variables.union(set(o[1] + o[2]))
        self.variables_ind = list(variables)

        self.indexed_ops = self.ops_to_indexed(ops)

        self.module_list = nn.ModuleList([n.module for n in node_list])
        self.variable_list = [Variable(requires_grad=True) for v in variables]

        # Find out the order of operations for reverse calculations
        ops_rev = []
        for i in self.ind_in:
            node_list[i].run_backward(ops_rev)
        self.indexed_ops_rev = self.ops_to_indexed(ops_rev)

    def ops_to_indexed(self, ops):
        '''Helper function to translate the list of variables (origin ID, channel),
        to variable IDs.'''
        result = []

        for o in ops:
            try:
                vars_in = [self.variables_ind.index(v) for v in o[1]]
            except ValueError:
                vars_in = -1

            vars_out = [self.variables_ind.index(v) for v in o[2]]

            # Collect input/output nodes in separate lists, but don't add to
            # indexed ops
            if o[0] in self.ind_out:
                self.return_vars.append(self.variables_ind.index(o[1][0]))
                continue
            if o[0] in self.ind_in:
                self.input_vars.append(self.variables_ind.index(o[1][0]))
                continue

            result.append((o[0], vars_in, vars_out))

        # Sort input/output variables so they correspond to initial node list
        # order
        self.return_vars.sort(key=lambda i: self.variables_ind[i][0])
        self.input_vars.sort(key=lambda i: self.variables_ind[i][0])

        return result

    def forward(self, x, rev=False):
        '''Forward or backward computation of the whole net.'''
        if rev:
            use_list = self.indexed_ops_rev
            input_vars, output_vars = self.return_vars, self.input_vars
        else:
            use_list = self.indexed_ops
            input_vars, output_vars = self.input_vars, self.return_vars

        if isinstance(x, (list, tuple)):
            assert len(x) == len(input_vars), (
                f"Got list of {len(x)} input tensors for "
                f"{'inverse' if rev else 'forward'} pass, but expected "
                f"{len(input_vars)}."
            )
            for i in range(len(input_vars)):
                self.variable_list[input_vars[i]] = x[i]
        else:
            assert len(input_vars) == 1, (f"Got single input tensor for "
                                          f"{'inverse' if rev else 'forward'} "
                                          f"pass, but expected list of "
                                          f"{len(input_vars)}.")
            self.variable_list[input_vars[0]] = x

        for o in use_list:
            try:
                results = self.module_list[o[0]]([self.variable_list[i]
                                                  for i in o[1]], rev=rev)
            except TypeError:
                raise RuntimeError("Are you sure all used Nodes are in the "
                                   "Node list?")
            for i, r in zip(o[2], results):
                self.variable_list[i] = r
            # self.variable_list[o[2][0]] = self.variable_list[o[1][0]]

        out = [self.variable_list[output_vars[i]]
               for i in range(len(output_vars))]
        if len(out) == 1:
            return out[0]
        else:
            return out

    def jacobian(self, x=None, rev=False, run_forward=True):
        '''Compute the jacobian determinant of the whole net.'''
        jacobian = 0

        if rev:
            use_list = self.indexed_ops_rev
        else:
            use_list = self.indexed_ops

        if run_forward:
            if x is None:
                raise RuntimeError("You need to provide an input if you want "
                                   "to run a forward pass")
            self.forward(x, rev=rev)
        jacobian_list = list()
        for o in use_list:
            try:
                node_jac = self.module_list[o[0]].jacobian(
                    [self.variable_list[i] for i in o[1]], rev=rev
                )
                jacobian += node_jac
                jacobian_list.append(jacobian)
            except TypeError:
                raise RuntimeError("Are you sure all used Nodes are in the "
                                   "Node list?")

        return jacobian
