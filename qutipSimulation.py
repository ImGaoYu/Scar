import enum
from qutip import *
import qutipEnvelopes as qenv
from pylab import *
import copy
'''
Bug fixed for python3
Shure Zhang 20191127
'''
class QuantumObj():
    def __init__(self, name, freqs, decoInfos={}):
        '''
        initialize the qobj.
        note that freqs=[f10, f21, ...] will be cumsummed to [f00,f10,f20,f30,...]'''
        #-------basic informations----
        self.freqs = np.hstack([[0],np.cumsum(freqs)])
        self.levels = len(self.freqs)
        self.decoInfos = decoInfos
        self.name = name
        #-------basic operators-------
        self.eye = qeye(self.levels)
        self.sm = destroy(self.levels)
        self.drvI = self.sm+self.sm.dag()
        self.drvQ = -1j*self.sm+1j*self.sm.dag()
        self.sigmax = np.eye(self.levels,dtype=complex)
        self.sigmax[:2,:2] = sigmax().full()
        self.sigmax = Qobj(self.sigmax)
        self.sigmay = np.eye(self.levels,dtype=complex)
        self.sigmay[:2,:2] = sigmay().full()
        self.sigmay = Qobj(self.sigmay)
        self.sigmaz = np.eye(self.levels,dtype=complex)
        self.sigmaz[:2,:2] = sigmaz().full()
        self.sigmaz = Qobj(self.sigmaz)
        
        self.states = self.state_ops()
        #-------sequences-------------
        self.xy = qenv.NOTHING
        self.z = qenv.NOTHING
        
    def state_ops(self):
        '''
            return projection operate for each states.
        '''
        states = []
        for idx,freq in enumerate(self.freqs):
            states.append(basis(self.levels,idx)*basis(self.levels,idx).dag())
        return states
    
    def rotate_op(self, theta, phi, angle):
        '''
        rotate the state along the shaft(theta,phi) for some angle.
        theta: angle between the rotation shaft and z axis.
        phi: angle between the xy projection of rotation shaft and x axis.
        angle: the rotation angle.
        '''
        op = -1j*angle/2.0*(np.sin(theta)*(np.cos(phi)*self.sigmax+np.sin(phi)*self.sigmay)+np.cos(theta)*self.sigmaz)
        return op.expm()
    
    def get(self, key, defaultVal=None):
        if hasattr(self,key):
            return getattr(self, key)
        else:
            return defaultVal
    
    def get_seq_func(self, key, conj=False):
        if hasattr(self, key):
            seq = getattr(self, key)
            def seqFunc(t, args):
                return seq(t, conj=conj)
        else:
            def seqFunc(t, args):
                return 0*t
        return seqFunc
    
    def get_state(self, amps):
        state = 0* basis(self.levels,0)
        for idx in range(len(amps)):
            state += amps[idx]*basis(self.levels,idx)
        return state.unit()
    
    def rotate(self, angle, theta, phi):
        '''rotate qubit state along the axis defined by [theta, phi] with angle'''
        op = np.cos(theta)*np.sin(phi)*sigmax()+np.sin(theta)*np.sin(phi)*sigmay()+np.cos(phi)*sigmaz()
        op = (-1j*angle/2*op).expm()
        op0 = op.full()
        op1 = np.eye(self.levels, dtype=complex)
        for idx1 in [0,1]:
            for idx2 in [0,1]:
                op1[idx1, idx2] = op0[idx1, idx2]
        
        return Qobj(op1)
        
    def test(self):
        print ('test')       
class QuantumSys():
    def __init__(self, qobjs, coupleInfos={}):
        self.coupleInfos = coupleInfos
        self.tlist = np.arange(0,100,0.1)
        self.state = tensor([qobj.states[0] for qobj in qobjs])#set to ground state by default
        self.qobjs = list(qobjs)
        self.levels = [qobj.levels for qobj in qobjs]
        self.names = [qobj.name for qobj in qobjs]
        self.eyes = [qobj.eye for qobj in qobjs]

        self.H = 0
    def find_index(self, qobj, state = 0):
        """
        find index where the qobj is in state
        
        parameters:
            qobj: quantumObj registered in system
            state: 0,1,2,...,level-1
        return:
            index
        """
        qidx = self.names.index(qobj.name)
        levelAll = np.prod(self.levels)
        order = np.arange(0,levelAll,np.prod(self.levels[qidx:]), dtype=np.int32).reshape(-1,1)
        q1 = int(np.prod(self.levels[qidx+1:]))
        order = order + np.arange(state*q1,(state+1)*q1).reshape(1,-1)   

        return order.reshape(-1)
    
    def add_qobj(self, qobj, state, coupleInfos={}):
        '''
            add a qobj in the quantum system.
        '''
        self.qobjs.append(qobj)
        self.state = tensor(self.state, state)
        self.coupleInfos.update(coupleInfos)
        self.levels = [qobj.levels for qobj in self.qobjs]
        self.names = [qobj.name for qobj in self.qobjs]
        self.eyes = [qobj.eye for qobj in self.qobjs]
    
    
    def append_qobj(self, qobj, coupleInfos={}):

        '''
            append a qobj in the quantum system.
        '''
        self.qobjs.append(qobj)
        self.state = tensor(self.state, qobj.states[0])
        self.coupleInfos.update(coupleInfos)
        self.levels = [qobj.levels for qobj in self.qobjs]
        self.names = [qobj.name for qobj in self.qobjs]
        self.eyes = [qobj.eye for qobj in self.qobjs]
    
    def add_coupleInfo(self, coupleInfos = {}):
        '''
            add couple informations
        '''
        self.coupleInfos.update(coupleInfos)
    
    def get_op(self, qobj, op):
        '''
            get single qubit operation for the system.
        '''
        qidx = self.names.index(qobj.name)
        op_list = copy.deepcopy(self.eyes)
        op_list[qidx] = op
        op = tensor(op_list)
        return op
    
    def get_ops(self, ops):
        '''
            generate operation for the sys.
            each term in ops correspond to a qobj with the same index in the system.
            ops can be a string or a list of operators.
            if ops is a string, each alphabet is an operation by the following map
        '''
        op_list = copy.deepcopy(self.eyes)
        if isinstance(ops, str):
            op_map = {'I':(0,0,0),'X/2':(np.pi/2,0,np.pi/2),'-X/2':(np.pi/2,np.pi,np.pi/2),'X':(np.pi/2,0,np.pi),
                       'Y/2':(np.pi/2,np.pi/2,np.pi/2),'-Y/2':(np.pi/2,-np.pi/2,np.pi/2),'Y':(np.pi/2,np.pi/2,np.pi),
                       'Z':(0,0,np.pi)}
            ops = ops.split(',')
            for idx, op in enumerate(ops):
                qobj = self.qobjs[idx]
                op_list[idx] = qobj.rotate_op(*op_map[op])
        else:
            for idx, op in enumerate(ops):
                qobj = self.qobjs[idx]
                op_list[idx] = qobj.rotate_op(op)
        op = tensor(op_list)
        return op

    def get_N(self, qobj):
        '''
            get N = create()*destroy() for single qubit
        '''
        qidx = self.names.index(qobj.name)
        op_list = copy.deepcopy(self.eyes)
        op_list[qidx] = create(2) * destroy(2)
        op = tensor(op_list)
        return op
     
    def apply_op(self, qobj, op, mode='normal'):
        ''' 
        mode: normal or fast
        apply operations on the system.
        update system state.        
        '''
        if mode is 'normal':
            op = self.get_op(qobj, op)
            self.state = op*self.state*op.dag()
        elif mode is 'fast':
            # 
            index0 = self.find_index(qobj, 0)
            index1 = self.find_index(qobj, 1)
            
            rotationOpt = op.full()
            state = self.state.full()
            state1 = np.copy(state)
            state1[index0] = rotationOpt[0,0]*self.state[index0] + rotationOpt[0,1]*state[index1]
            state1[index1] = rotationOpt[1,0]*state[index0] + rotationOpt[1,1]*state[index1]
            self.state = Qobj(state1, dims=[self.levels, [1]*len(self.levels)])
        return self.state
    def get_model(self,improved=False):
        '''
        construct Model.
        This is done by adding energy term, coupling term and driving term.
        Improved option means counter routating term.
        '''
        H0 = 0
        Ht = []
        
        for idx0, qobj in enumerate(self.qobjs):
            sm = self.get_op(qobj, qobj.sm)
            
            if not qobj.xy == qenv.NOTHING:
                xyFunc = qobj.get_seq_func('xy',True)
                xyFunc_dag = qobj.get_seq_func('xy',False)
                Ht.extend([[sm, xyFunc],[sm.dag(), xyFunc_dag]])
          
            for idx1, freq in enumerate(qobj.freqs):
                state = self.get_op(qobj, qobj.states[idx1])
                H0 += 2 * np.pi * freq * state
                if not qobj.z == qenv.NOTHING:
                    z = qobj.get_seq_func('z')
                    if idx1 != 0:
                        Ht.append([state, z])
                     
                
                
        for coupleInfo in self.coupleInfos.keys():
            qNames = coupleInfo.split('-')
            qidxs = [self.names.index(qName) for qName in qNames]
            g = self.coupleInfos[coupleInfo]
            qs = [self.qobjs[qidx] for qidx in qidxs]
            sms = [self.get_op(qobj, qobj.sm) for qobj in qs]
            H0 += 2 * np.pi * g * (sms[0] * sms[1].dag() + sms[0].dag() * sms[1])
            if improved:
                H0 += -2 * np.pi * g * (sms[0].dag() * sms[1].dag() + sms[0] * sms[1])
        self.H = H0 if Ht == [] else [H0] + Ht
       
        return 
            
    def get_c_op_list(self):
        '''
        get collapes operators for decoherence process
        '''
        c_op_list = []
        for idx0, qobj in enumerate(self.qobjs):
            decoInfos = qobj.decoInfos
            sm = self.get_op(qobj,qobj.sm)
            
            for key in decoInfos.keys():
                if key == 'T1':
                    c_op_list.append(np.sqrt(1./decoInfos[key])*sm)
                if key == 'T2':
                    c_op_list.append(np.sqrt(2.0/decoInfos[key])*sm.dag()*sm)
        return c_op_list
    
    def extract_dm(self, qobjs):
        '''extract density matrix'''
        ptraceIndexs = []
        for idx, qobj in enumerate(qobjs):
            qidx = self.names.index(qobj.name)
            ptraceIndexs.append(qidx)
        print ('ptracing...', ptraceIndexs)
        rho = ptrace(self.state, ptraceIndexs)
        return rho
        
    def extract_dm_all(self, qobjs, results):
        '''extract all density matrices'''
        ptraceIndexs = []
        for idx, qobj in enumerate(qobjs):
            qidx = self.names.index(qobj.name)
            ptraceIndexs.append(qidx)
        print ('ptracing...', ptraceIndexs)
        rho = [ptrace(result, ptraceIndexs) for result in results]
        return rho
    def get_eigenstates(self):
        '''derive eigenstates of the system'''
        self.eigenV =[]
        self.eigenS = []
        Hs = np.array([self.H[0].full()] * len(self.tlist))
        dims = self.H[0].dims
        shape = self.H[0].shape
        if type(self.H) is list:
            for ht in self.H[1:]:
                Hs += np.array(np.array([ht[0].full()] * len(self.tlist))* (ht[1](self.tlist,False)).reshape(len(self.tlist),1,1))
            for h in Hs:
                h = Qobj(h,dims =dims,shape=shape)
                ev,es = h.eigenstates()
                self.eigenV.append(ev/2/np.pi)
                self.eigenS.append(es)
        self.eigenV = np.array(self.eigenV)
        self.eigenS = np.array(self.eigenS)
    def u_evo(self, decoherence=True,improved=True):
        '''U for qpt or gate process'''
        if self.H == 0:
            self.get_model(improved)
        print('Hamiltonian generated')
        if decoherence:
            c_op_list = self.get_c_op_list()
        else:
            c_op_list = []
        print('Master Equation sovling')
        U = propagator(self.H, self.tlist, self.get_c_op_list())       
        return U
    def state_evo(self, update=False, decoherence=True,bar=True,improved=False,do_chop=True,index=[]):
        '''if update, then update the final state to self.state'''
        if self.H == 0:
            self.get_model(improved)
        print('Hamiltonian generated')
        if decoherence:
            c_op_list = self.get_c_op_list()
            if do_chop:
                for idx, c_op in enumerate(c_op_list):
                    c_op_list[idx] = Qobj(c_op.data[:,index][index,:])
        else:
            c_op_list = []
        print('Master Equation sovling')
        options = Options(nsteps=2000)
        options.atol = 1e-8
        options.rtol = 1e-8
        result = mesolve(self.H, self.state, self.tlist, c_op_list, [], options=options, progress_bar=bar).states       
        if update:
            self.state = result[-1]
        return result
