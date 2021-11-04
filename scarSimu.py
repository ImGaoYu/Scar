import time, math
from qutip import * 
from qutip.qip.gates import rx,ry
import numpy as np
import matplotlib.pyplot as plt
import qutipSimulation as qs
import copy
import itertools
import random
from scipy.optimize import leastsq, fsolve, curve_fit
from scipy.signal import find_peaks
import scipy.io as scio
import datetime
import matplotlib.colors as colors
import matplotlib.cm as cm
plt.ion()
def fft_privite(xs, ys):
    freq = np.fft.fftfreq(len(xs), xs[1]-xs[0])
    fourier = abs(np.fft.fft(ys, len(xs)))
    freq = freq[1:]
    g = np.abs(freq[np.argmax(fourier[1:])])
    return round(g, 2)
    
    
    
def qqSwapSE(freq=6.5, delta=-1.0, improved=False, do_plot = 1):
    eta = -0.26
    etac = -0.35
    freqc = freq - delta
    q1 = qs.QuantumObj('q1',[freq])
    q2 = qs.QuantumObj('q2',[freq])
    c = qs.QuantumObj('c',[freqc])
    state = tensor(q1.get_state([0,1]),q2.get_state([1]),c.get_state([1]))

    tlist = np.arange(0,1000,0.2)
    coupleInfos = {'q1-c':0.13,'q2-c':0.13, 'q1-q2':0.005}
    
    sys = qs.QuantumSys([q1,q2,c], coupleInfos=coupleInfos)
    sys.tlist = tlist
    sys.state = state
    results = sys.state_evo(decoherence = False, improved = improved)
    
    op1 = sys.get_op(q1, ket2dm(q1.get_state([0,1])))
    op2 = sys.get_op(q2, ket2dm(q2.get_state([0,1])))
    P1 = expect(op1, results)
    P2 = expect(op2, results)
    
    if do_plot:
        plt.figure()
        plt.plot(tlist, P1, '.-', label='Prob_q1')
        plt.plot(tlist, P2, '.-', label='Prob_q2')
        plt.xlabel('time (ns)')
        plt.ylabel('P1')
    
    return P1
    

def qqSwapScan(freq = 6.5, improved = False):
    deltaList = np.linspace(-0.1,-5,60)
    tlist = np.arange(0,1000,0.2)
    Fd = np.zeros((len(deltaList),len(tlist)))
    for i in range(len(deltaList)):
        delta = deltaList[i]
        Fd[i,:] = qqSwapSE(freq, delta, improved, do_plot=0)
    plt.figure()
    plt.pcolor(tlist,deltaList,Fd)
    plt.xlabel("time(ns)", size = 20)
    plt.ylabel("Freq of Coupler", size = 20)
    return Fd

def get_N(qnum, cnum, idx):
    op_nList = [identity(2) for i in range(qnum+cnum)]
    op_nList[idx] = create(2) * destroy(2)
    op_N = tensor(op_nList)
    return op_N

def genHchopIndex(qnum,cnum, n_ph):
    #TODO: fewer photons case
    # combins = itertools.combinations(range(qnum+cnum), int(qnum/2))
    combins = itertools.combinations(range(qnum+cnum), n_ph)
    index = np.sort([np.sum(2**(qnum+cnum-1-np.asarray(combin))) for combin in combins])
    return index

def genCoupleInfos(qnum, cnum, qubits, couplers, Xtalk, modelType = 'comb', gA=15, gB=10, g_qc=80):
    '''
    Generate coupling informations - Comb model
    '''
    coupleInfos = {}
    if modelType is 'comb':
        temp = 0
    elif modelType is 'SSH':
        temp = -1
    
    # Xtalk
    if Xtalk:
        for i in range(0, qnum, 1):
            for j in range(i+1,qnum,1):
                link = qubits[i].name + '-' + qubits[j].name
                coupleInfos.update({link:0.005/(j-i)*random.random()})
    
    #TODO: load the coupling information load experiment chips
    # if modelType == 'comb':
    #     # NN coupling of comb
    #     NNcouplingList = [[0,2],[1,5],[2,5],[3,4],[3,7],[5,6]]
    # elif modelType == 'SSH':
    #     SSHcouplingList = [[1,3],[2,4],[3,7],[4,6],[4,8],[5,7],[7,9],[9,11],[11,13]]
    #     NNcouplingList = []
    #     for coupling in SSHcouplingList:
    #         if coupling[0] < qnum and coupling[1] < qnum:
    #             NNcouplingList.append(coupling)
    # for coupling in NNcouplingList:
    #     link = qubits[coupling[0]].name + '-' + qubits[coupling[1]].name
    #     coupleInfos.update({link:0.001+0.00*random.random()})
    
    # Direct coupling
    if cnum:
        for i in range(0,qnum,2):
            link = qubits[i].name + '-' + qubits[i+1].name
            coupleInfos.update({link:0.005})
        for i in range(1,qnum-2,2):
            link = qubits[i].name + '-' + qubits[i+2+temp].name
            coupleInfos.update({link:0.005})
    else:
        for i in range(0,qnum,2):
            link = qubits[i].name + '-' + qubits[i+1].name
            coupleInfos.update({link:gA/1000})
        for i in range(1,qnum-2,2):
            link = qubits[i].name + '-' + qubits[i+2+temp].name
            coupleInfos.update({link:gB/1000})
    # QC coupling
    for i in range(cnum):
        link1 = qubits[i].name + '-' + couplers[i].name
        if i%2 == 0:
            link2 = qubits[i+1].name + '-' + couplers[i].name
        else:
            link2 = qubits[i+2+temp].name + '-' + couplers[i].name
        coupleInfos.update({link1:g_qc/1000})
        coupleInfos.update({link2:g_qc/1000})
    
    
    return coupleInfos

def genCoupleInfos_2D(qnum, cnum, qubits, couplers, Xtalk, modelType = 'comb'):
    '''
    Generate 2D lattice coupling informations
    '''
    coupleInfos = {}
    
    # Xtalk
    if Xtalk:
        for i in range(0, qnum, 1):
            for j in range(i+1,qnum,1):
                link = qubits[i].name + '-' + qubits[j].name
                coupleInfos.update({link:0.005/(j-i)*random.random()})
    
    # Direct coupling
    """Firstly construct the 2D lattice"""
    for i in range(4):
        for j in np.arange(i*4,i*4+3):
            link = qubits[j].name + '-' + qubits[j+1].name
            coupleInfos.update({link:0.01})
        for j in np.arange(i, qnum-4, 4):
            link = qubits[j].name + '-' + qubits[j+4].name
            coupleInfos.update({link:0.01})
    if modelType == '2D':
        for i in range(4):
            for j in np.arange(i*4,i*4+3):
                link = qubits[j].name + '-' + qubits[j+1].name
                coupleInfos.update({link:0.02})
    elif modelType == '2D2':
        """Considering other structures"""
        # Ridx = [0,2,5,9,12,14] # index of strong coupling in rows
        # Cidx = [1,2,4,7,9,10] # index of strong coupling in rows
        Ridx = [0,2,4,6,8,10,12,14] # index of strong coupling in rows
        Cidx = [0,1,2,3,8,9,10,11] # index of strong coupling in rows
        for j in Ridx:
            link = qubits[j].name + '-' + qubits[j+1].name
            coupleInfos.update({link:0.02})
        for j in Cidx:
            link = qubits[j].name + '-' + qubits[j+4].name
            coupleInfos.update({link:0.02})
    
    return coupleInfos

def getR(qnum, qubits, couplers, modelType='comb', Xtalk=0, gA=15, gB=10):
    coupleInfos2 = genCoupleInfos(qnum, 0, qubits, couplers, Xtalk=Xtalk, modelType=modelType, gA=gA, gB=gB)
    sys2 = qs.QuantumSys(qubits, coupleInfos=coupleInfos2)
    # Hmatrix = sys.H.full()
    # eigVals = np.linalg.eigvals(Hmatrix)
    sys2.get_model(improved=1)
    eigVals = sys2.H.eigenenergies()
    eigVals = np.sort(np.real(eigVals.data))
    delta = np.diff(eigVals)
    # plt.figure()
    # plt.hist(delta)
    R = [np.min([delta[ii], delta[ii+1]])/np.max([delta[ii], delta[ii+1]]) for ii in range(len(delta)-1)]
    for idx, Ri in enumerate(R):
        if Ri < 0.001 or np.isnan(Ri):
            R[idx] = 0.0
    return np.average(R)
    
def getIniState(qnum, sys, thermal=0, modelType='comb', idx = []):
    op_list = [basis(2,0) for qobj in sys.qobjs]
    stateQ = ''
    if thermal:
        if idx == []:
            print("Now simulating thermal states......")
            combins = itertools.combinations(range(qnum), int(qnum/2))  # int(qnum/2)
            combins = np.asarray([np.asarray(combin) for combin in combins])
            ind = random.randint(0,len(combins)-1)
            for i in combins[ind]:
                op_list[i] = basis(2,1)
        else:
            '''It can be used on the case of fewer photons'''
            for i in idx:
                op_list[i] = basis(2,1)
    elif modelType is 'SSH':
        for i in range(0,qnum,4):
            op_list[i]=basis(2,1)
        for i in range(3,qnum,4):
            op_list[i]=basis(2,1)
    elif modelType is 'comb':
        # op_list[0] = basis(2,1)
        for i in range(0,qnum,2):
            op_list[i] = basis(2,1)
    elif modelType[0:2] == '2D':
        if idx == []:
            idx = [0,3,5,6,8,11,13,14]
        for i in idx:
            op_list[i] = basis(2,1)
    else:
        print("Wrong modelType!")
    print("*"*20)
    print("The psi0 List is:")
    for i in range(qnum):
            if op_list[i] == basis(2,1):
                stateQ += '1'
            else:
                stateQ += '0'
    print(stateQ)
    return tensor(op_list), stateQ

#TODO: Simplify the code of entanglement entropy
def getEE(qnum, cnum, phi):
    rho = ket2dm(phi)
    rho_A = rho.ptrace(np.arange(4)) #[i+int(qnum/2) for i in range(int(qnum/2))]
    S_A = entropy_vn(rho_A)
    return S_A

def getBasis(qnum, cnum, representation = 'Hilbert'):
    '''
    get the probability basis such as:
    1 1 1 1 0 0 0 0 
    1 1 1 0 1 0 0 0
        ......
    0 0 0 0 1 1 1 1
    '''
    combins = itertools.combinations(range(qnum+cnum), int(qnum/2))
    Basis = []
    for combin in combins:
        n_i = [0 for i in range(qnum)]
        for i in combin:
            n_i[i] = 1
        Basis.append(n_i)
    return Basis

def getBasisEE(qnum, cnum):
    combins = itertools.combinations(range(qnum+cnum), int(qnum/2))
    phiList = []
    for combin in combins:
        phi_i = [basis(2,0) for i in range(qnum)]
        for i in combin:
            phi_i[i] = basis(2,1)
        for i in range(int(qnum/2)):
            phi_i[i] = qeye(2)
        phiList.append(phi_i)
    return phiList

def getEE2(qnum, cnum, phi):
    n_ph = int(qnum/2)
    index = genHchopIndex(qnum, cnum, n_ph = n_ph)
    phi = np.array([phi])
    rho = phi * np.transpose(phi)
    phiList = getBasisEE(qnum, cnum)
    # rho_A = tensor([qzero(2) for i in range(int(qnum/2))])
    # rho_A = rho_A.full()[:,0:len(rho)][0:len(rho),:]
    rho_A = []
    start0 = time.time()
    for phi_i in phiList:
        phi_i = tensor(phi_i).full()[index]
        rho_A .append(np.dot(np.dot(np.transpose(phi_i), rho), phi_i))
    rho_A = sum(rho_A)
    start1 = time.time()
    print(f'=======for loop: {start1-start0}')
    Value = np.linalg.eigvals(rho_A)
    start2 = time.time()
    print(f'=======eigenValues: {start2-start1}')
    S_A = -np.dot(Value, np.log(Value))
    return S_A

def plotEE(qnum, cnum, sys):
    '''Get eigenvalues and eigenstates, then plot the EE of system'''
    '''Animate the process'''
    V, S = np.linalg.eig(sys.H)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Eigen Energy')
    ax.set_ylabel('Entanglement Entropy')
    ax.set_title('EE of Eigenstates')
    phi = S[0]
    S_A = getEE(qnum, cnum, phi)
    line = ax.plot([V[0], V[0]], [S_A, S_A], marker='o')[0]
    plt.ion()
    dataS = [S_A]
    dataV = [V[0]]
    for idx, Vi in enumerate(V):
        phi = S[idx]
        dataS.append(getEE(qnum, cnum, phi))
        dataV.append(Vi)
        line.set_xdata(dataV)
        line.set_ydata(dataS)
        ax.set_xlim([V[0]-2, V[-1]+2])
        ax.set_ylim([-1, 6])
        plt.pause(0.001)
        print(f'{idx}th eigenenergy is {Vi}, S={dataS[-1]}')
    return dataS

def plotEE2(qnum, cnum, sys):
    '''If do_chop is True, get eigenvalues and eigenstates, then plot the EE of system'''
    # V, S = np.linalg.eig(sys.H)  # When do_chop is True, we use this to diagonalize
    V, S = sys.H.eigenstates()
    dataS = []
    dataV = []
    for idx, Vi in enumerate(V):
        phi = S[idx]
        dataS.append(getEE(qnum, cnum, phi))
        print(f'{idx}th eigenenergy is {Vi}, S={dataS[-1]}')
    dataV = V
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Eigen Energy')
    ax.set_ylabel('Entanglement Entropy')
    ax.set_title(f'EE of {qnum}qubits Eigenstates')
    plt.plot(dataV, dataS, 'o')
    return dataS

def plotEE3(qnum, cnum, sys):
    '''If do_chop is False, get eigenvalues and eigenstates, then plot the EE of system'''
    # V, S = np.linalg.eig(sys.H)  # When do_chop is True, we use this to diagonalize
    V, S = sys.H.eigenstates()
    '''Each photon in qubits has energy about 40'''
    # index = np.where((V > (40*qnum/2 - 10)) & (V < (40*qnum/2 +10)))
    dataS = []
    dataV = []
    # V = V[index]
    for idx, Vi in enumerate(V):
        phi = S[idx]
        Si = getEE(qnum, cnum, phi)
        dataS.append(Si)
        print(f'{idx}th eigenenergy is {Vi}, S={dataS[-1]}')
    dataV = V
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Eigen Energy')
    ax.set_ylabel('Entanglement Entropy')
    ax.set_title(f'EE of {qnum}qubits Eigenstates')
    plt.plot(dataV, dataS, 'o')
    return dataS

def ScarRun(qnum, cnum = 1, freq = 6.8, gA = 15, gB = 10, modelType = 'comb', 
            idx = [], fig = None, impurity = None, thermal = False, do_plot = True, 
            do_save = False, decoherence = False, do_chop = True, Xtalk = False, g_qc=80, 
            improved = True, tlist = None, doEE=False, doScatter=False, savepath = ''):
    if cnum:
        print("Considering the couplers......")
        cnum = qnum - 1
    delta1 = CalcDelta(g_qc=g_qc,g_eff=gA)
    delta2 = CalcDelta(g_qc=g_qc,g_eff=gB)
    #delta1 = -0.865, delta2 = -1.2
    freqc1 = freq - delta1
    freqc2 = freq - delta2
    if decoherence:
        decoInfos={ 'T1':1000, 'T2':1000}
    else:
        decoInfos={}
    
    if impurity == None:
        impurity = [0.0] * qnum
    elif isinstance(impurity,float):
        impurity = [impurity] * 2 + [0.0] * (qnum - 2)
    elif isinstance(impurity, list or np.ndarray) and len(impurity) <= qnum:
        impurity = np.hstack([impurity, [0.0] * (qnum - len(impurity))])
    else:
        print("Wrong impurity!")
        return
    qubits = [qs.QuantumObj('q%d'%(i+1),[freq-impurity[i]], decoInfos) for i in range(qnum)]
    couplers = [qs.QuantumObj('c%d'%(i+1),[freqc1]) for i in range(cnum)]

    for i in range(1,cnum,2):
        couplers[i].freqs[1] = freqc2

    #TODO: In the rotating frame, the R is not calculated correctly

    R = getR(qnum, qubits, couplers, modelType, Xtalk, gA=gA, gB=gB)
    print("*"*20)
    print("The value of <R> is:",R)

    if modelType == 'comb' or modelType == 'SSH':
        coupleInfos = genCoupleInfos(qnum, cnum, qubits, couplers, Xtalk, 
                            modelType = modelType, gA = gA, gB = gB, g_qc = g_qc)
    else:
        coupleInfos = genCoupleInfos_2D(qnum, cnum, qubits, couplers, Xtalk, modelType=modelType)

    sys = qs.QuantumSys(qubits+couplers, coupleInfos=coupleInfos)

    if tlist is None:
        tlist = np.arange(0,200,1.0)
    sys.tlist = tlist

    state, stateQ = getIniState(qnum, sys, thermal=thermal, modelType=modelType, idx=idx)
    # state = 1/np.sqrt(2)*(tensor(basis(2,0),basis(2,1)) - tensor(basis(2,1),basis(2,0)))
    sys.state = state
    sys.get_model(improved)
    # return sys.H
    

#TODO: modify the chop code
    if do_chop:
        n_ph = int(qnum/2)
        index = genHchopIndex(qnum, cnum, n_ph = n_ph)
        state = Qobj(state.full()[index])
        sys.state = state
        sys.H = Qobj(sys.H.data[:,index][index,:]) 
    # else:
    #     index = np.array([i for i,_ in enumerate(state)])
    # state = Qobj(state.full()[index])
    # sys.H = Qobj(sys.H.data[:,index][index,:]) 
    # return sys.H
    
    
    if doEE:
        dataS = plotEE3(qnum, cnum, sys)
        return sys, dataS
    #TODO: plot the scatter of energies
    # if doScatter:
    #     scatterEnergy(sys.H)

    results = sys.state_evo(decoherence=decoherence, improved = improved, do_chop=do_chop, index=index)
    # return results, sys.H
    P1 = expect(state*state.dag(), results)
    popu_t = []
    if do_chop:
        for i in range(qnum):
            popu_t.append(expect(Qobj(get_N(qnum,cnum,i).data[:,index][index,:]), results))
    else:
        for i in range(qnum):
            popu_t.append(expect(Qobj(get_N(qnum,cnum,i)), results))
    popu_t = np.asarray(popu_t)

    if do_plot:
        # if impurity:    titleName = modelType + ' with impurity ' +str(impurity*1000) + 'MHz'
        # else:   titleName = modelType + ' no impurity'
        # if thermal:     titleName = titleName + ' Ther'
        Fid = pltImbalance(qnum, cnum, stateQ, popu_t, P1, tlist, modelType=modelType, fig=fig)
        plotCoupleInfos(qnum, coupleInfos, modelType, stateQ)
    else:
        Fid = calcImbalance(qnum, stateQ, popu_t)
                
    if do_save:
        if savepath == '':
            savepath = 'D:/Share/GY/QuantumManybodyScar/20210927Figure/VaryCoupling/'
        if fig == None:
            sfig = ''
        else:
            sfig = str(fig)
        saveName = modelType + "%dq%dc"%(qnum,cnum) + stateQ + sfig + ".mat"
        scio.savemat(savepath+saveName, {'tlist':tlist, 'Fid':P1, 'population':popu_t, 'Imbalance':Fid, 'coupleInfos':coupleInfos, 'impurity': impurity})
    
    return results, sys.H, stateQ, popu_t, Fid, P1, R

def plotCoupleInfos(qnum, coupleInfos, modelType, stateQ):
    """Draw the illustration of the system structure"""
    # Generate the coordinate
    #TODO: finish other modelType
    xCord = []
    yCord = []
    if modelType[0:2] == '2D':
        for i in range(4):
            xCord = np.hstack([xCord, np.array([1,2,3,4])])
            yCord = np.hstack([yCord, np.array([4-i]*4)])
    elif modelType == 'comb' or modelType == 'SSH':
        xCord = np.reshape([[i,i] for i in range(int(qnum/2))], [1,qnum]).flatten()
        yCord = np.array([1,0]*int(qnum))
    fig = plt.figure()
    ax = plt.subplot(111)
    # Generate the couplings
    for coupleInfo in coupleInfos.keys():
        qNames = coupleInfo.split('-')
        qNames = [int(qNames[i][1:])-1 for i in range(len(qNames))]
        print(qNames)
        print(coupleInfos[coupleInfo])
        lcolor = 'navy' if coupleInfos[coupleInfo] > 0.01 else 'cyan'
        ax.plot([xCord[qNames[0]], xCord[qNames[1]]], [yCord[qNames[0]],yCord[qNames[1]]],
                     '-', color=lcolor, linewidth=5)
    ax.axis('off')
    for idx, value in enumerate(stateQ):
        marker = 'ro' if value == '1' else 'bo'
        ax.plot(xCord[idx],yCord[idx],marker,markersize=20)

def plotPhotons(qnum, cnum, n_ph):
    if cnum:
        cnum = qnum - 1
    combins = itertools.combinations(range(qnum), n_ph)
    freq = []
    yfft = []
    time=100
    for combin in combins:
        idx = np.asarray(combin)
        result, H = ScarRun(qnum, cnum,  modelType='comb', idx = idx,impurity=0.0,thermal=1,do_chop = False, time=time)
        P = [] 
        if n_ph is 2:
            sigList = [[sigmax(),sigmax()],[sigmay(),sigmay()],[sigmax(),sigmay()],[sigmay(),sigmax()]]
        elif n_ph == 1:
            sigList = [[sigmax()],[sigmay()]]
        for j in range(len(sigList)):
            op_List = [qeye(2) for i in range(qnum+cnum)]
            for n,i in enumerate(idx):
                op_List[i] = sigList[j][n]
            P.append(expect(tensor(op_List), result))
        if n_ph == 2:
            data = P[0] - P[1] + 1j * P[2] + 1j * P[3]
        elif n_ph == 1:
            data = P[0] + 1j * P[1]
        tlist = np.arange(0,time,0.01)
        plt.figure(5)
        plt.figure()
        plt.plot(tlist, data)
        freqi, yffti = fft(tlist, data)
        freq.append(freqi)
        yfft.append(yffti)
        plt.figure(4)
        plt.plot(freqi, yffti)
    plt.figure(110)
    plt.plot(freq[0], sum(yfft,0))
    plt.close(5)
    # return freq[0], sum(yfft,0), H

def CampareE(qnum, cnum, n_ph, freq_off, freq, yfft, H):
    # freq, yfft, H = plotPhotons(qnum,cnum, n_ph)
    index = genHchopIndex(qnum, cnum, n_ph)
    H = H[:,index][index,:]
    V, S = np.linalg.eig(H)
    arg = np.argsort(V)
    V1 = V[arg]
    combins = itertools.combinations(range(qnum), n_ph)
    combins = np.asarray([np.asarray(combin) for combin in combins])
    length = len(combins)
    plt.figure()
    for m in V1[:length]-freq_off:
        plt.vlines(m, 0, 1,color = 'red', linestyles='dashed')
    # plt.plot(V1[:length],[1 for i in range(len(V1[:length]))],'o', )
    plt.plot(freq, yfft)


def plotPhotons2(qnum, cnum, n_ph):
    if cnum:
        cnum = qnum - 1
    combins = itertools.combinations(range(qnum), n_ph)
    freq = []
    yfft = []
    time=100
    for combin in combins:
        idx = np.asarray(combin)
        result = ScarRun(qnum, cnum,  modelType='comb', idx = idx,impurity=0.0,thermal=1,do_chop = False, time=time)
        P = []
        if n_ph is 2:
            sigList = [[sigmax(),sigmax()],[sigmay(),sigmay()],[sigmax(),sigmay()],[sigmay(),sigmax()]]
        elif n_ph == 1:
            sigList = [[sigmax()],[sigmay()]]
        for j in range(len(sigList)):
            op_List = [qeye(2) for i in range(qnum+cnum)]
            for n,i in enumerate(idx):
                op_List[i] = sigList[j][n]
            P.append(expect(tensor(op_List), result))
        if n_ph == 2:
            data = P[0] - P[1] + 1j * P[2] + 1j * P[3]
        elif n_ph == 1:
            data = P[0] + 1j * P[1]
        tlist = np.arange(0,time,0.01)
        plt.figure()
        plt.plot(tlist, data)
        freqi, yffti = fft(tlist, data)
        freq.append(freqi)
        yfft.append(yffti)
        plt.figure(4)
        plt.plot(freqi, yffti)
    plt.figure(110)
    plt.plot(freq[0], sum(yfft,0))

def fft(xs, ys, doPlot=False, des=''):
    x_step = xs[1] - xs[0]
    x_max = max(xs)
    x_extend = np.arange(x_max+x_step, 300, x_step)
    y_extend = [0.0]*len(x_extend)
    xs = list(xs) + list(x_extend)
    ys = list(ys) + list(y_extend)
    freq = np.fft.fftfreq(len(xs), xs[1]-xs[0])
    fourier = abs(np.fft.fft(ys, len(xs)))/len(xs)
    arg = np.argsort(freq)
    freq = freq[arg]
    fourier = fourier[arg]
    if doPlot:
        plt.figure()
        plt.plot(freq, fourier, 's-', markersize=5)
        plt.title(des)
    return freq, fourier

def EvolveEE(qnum, cnum, tlist=None, thermal=0, do_plot = True, do_save=False, modelType='comb', idx=[], decoherence=True):
    if tlist is None:
        tlist = np.arange(0,1000,0.5)
    results, H, stateQ = ScarRun(qnum, cnum, freq=6.65, modelType=modelType, impurity=0.0, thermal=thermal, 
    do_chop=False,tlist=tlist, do_plot=do_plot, decoherence=decoherence, idx=idx)
    Stlist = np.zeros(len(tlist))
    SigmaX = []
    # return results
    for it, state in enumerate(results):
        print(it)
        # SigmaX.append(expect(get_N(qnum,cnum,0), state))
        Si = getEE(qnum, cnum, state)
        Stlist[it] = Si
    if do_save:
        a = datetime.datetime.now()
        tnow = a.strftime("%Y%m%d_%H%M")
        states = []
        for i, state in enumerate(results):
            states.append(np.asarray(state))
        states = np.array(states)
        scio.savemat(tnow+'_'+stateQ+"EE.mat",{"states":states, "Stlist":Stlist})
    if do_plot:
        plt.figure(20)
        plt.plot(tlist, Stlist, 's-')
        plt.title(f"{qnum}qubits{cnum}couplers{thermal}",fontsize=15)
        plt.xlabel("time(ns)",fontsize=20)
        plt.ylabel("Entanglement Entropy", fontsize=20)
        plt.tight_layout()
    return results, Stlist, SigmaX

def TuneCoupler(qnum, start=15, stop=5, scanNum=10, time=200):
    Fid = []
    tlist = np.arange(0,time,0.01)
    gList = np.linspace(start,stop,scanNum)
    for i in gList:
        delta1 = CalcDelta(g_eff = i)
        delta2 = CalcDelta(g_eff = i/1.5)
        qnumScan(delta1=delta1,delta2=delta2)
        # Fid.append(ScarRun(qnum,delta1=delta1,delta2=delta2,do_plot=1,do_save=1, time=time)[0])
    if 0:
        plt.figure()
        xx,yy = np.meshgrid(tlist, [i for i in gList])
        plt.pcolor(xx,yy, Fid)
        plt.ylabel("Freq of Couplers", size = 20)
        plt.colorbar()
        plt.tight_layout()
    if 0:
        scio.savemat("CouplerTuner.mat", {'Fid':Fid})
    return Fid

def CouplerScan(qnum, scanRange1=[0,-4], scanRange2=[0,-4], scanNum=20, modelType="comb"):
    tlist = np.arange(0,800,0.01)
    couplerFreq1 = np.linspace(scanRange1[0]+0.865,scanRange1[1],scanNum)
    couplerFreq2 = np.linspace(scanRange2[0]+0.865,scanRange2[1],scanNum)
    Peaks = []
    Peaks_freq = []
    Peaks_sum = []
    Fidelity = []
    r_ave = []
    population = []
    for coupler1 in couplerFreq1:
        for coupler2 in couplerFreq2:
            Fid, popu_t, R = ScarRun(qnum, delta1=-coupler1, delta2=-coupler2, do_plot=False, modelType = modelType)
            distance = int(1e9/(130*130/coupler1)/1e6/2*100)
            print(distance)
            Fidelity.append(Fid)
            population.append(popu_t)
            r_ave.append(R)
            peak_id, peak_property = find_peaks(Fid, height=0.05, distance=distance)
            Peaks_freq.append(tlist[peak_id])
            Peaks.append(peak_property['peak_heights'])
            Peaks_sum.append(sum(peak_property['peak_heights']))
    Peaks_sum = np.reshape(Peaks_sum,(scanNum,scanNum))
    r_ave = np.reshape(r_ave,(scanNum,scanNum))
    saveName = modelType + "CouplerScan"
    scio.savemat(saveName, {'Fid':Fidelity, 'population':population, 'R':r_ave, 'Peaks_freq':Peaks_freq,
                            'Peaks':Peaks, 'Peaks_sum':Peaks_sum})
    if 0:
        plt.figure()
        xx,yy = np.meshgrid(couplerFreq1, couplerFreq2)
        plt.pcolor(xx,yy,Peaks_sum)
        plt.ylabel("Freq of Coupler1", size = 20)
        plt.ylabel("Freq of Coupler2", size = 20)
        plt.colorbar()
        plt.tight_layout()
        plt.figure()
        plt.pcolor(xx,yy,r_ave)
        plt.ylabel("Freq of Coupler1", size = 20)
        plt.ylabel("Freq of Coupler2", size = 20)
        plt.colorbar()
        plt.tight_layout()

    return Fidelity, Peaks, Peaks_freq, Peaks_sum, r_ave

def pltFidelity(qnum, cnum, Fid, popu_t, tlist, titleName='Fidelity'):
    plt.figure()
    plt.subplot(211)
    plt.plot(tlist, Fid, 'o', label='Prob_q1')
    plt.xlabel('time (ns)', size = 20)
    plt.ylabel("Fidelity", size = 20)
    plt.title(titleName, size = 20)
    plt.subplot(212)
    xx,yy = np.meshgrid(tlist, [i for i in range(qnum+1)])
    plt.pcolor(xx,yy, popu_t)
    plt.ylabel("Qubits", size = 20)
    plt.colorbar()
    plt.tight_layout()

def calcImbalance(qnum, stateQ, popu_t):
    if popu_t.ndim == 2:
        psi0 = np.array([0]*qnum)
        for i,s in enumerate(stateQ):
            if s is '1':
                psi0[i] = 1
        Fid = np.dot(psi0, popu_t)/(qnum/2)
    elif popu_t.ndim == 1:
        Fid = popu_t
    return Fid

def pltImbalance(qnum, cnum, stateQ, popu_t, Fid, tlist, titleName='Imbalance', modelType='comb', fig = None):
    Imbalance = calcImbalance(qnum, stateQ, popu_t)
    plt.figure(fig,figsize=(10,15))
    plt.subplot(211)
    plt.plot(tlist, Imbalance, 'o-', label='{qnum}qubits')
    plt.ylim([0,1])
    plt.xlabel('time (ns)', size = 20)
    plt.ylabel("Imbalance", size = 20)
    # plt.xlim([0,100])
    # plt.title(titleName, size = 20)

    # plt.subplot(312)
    # plt.plot(tlist, Fid, 'o-', label='Prob_q1')
    # plt.xlabel('time (ns)', size = 20)
    # plt.ylabel("Fidelity", size = 20)

    plt.subplot(212)
    xx,yy = np.meshgrid(tlist, [i for i in range(qnum+1)])
    plt.pcolor(xx,yy, popu_t)
    plt.ylabel("Qubits", size = 20)
    plt.colorbar()
    plt.suptitle(f"{stateQ}",size=20)
    plt.tight_layout()
    return Imbalance

def CalcDelta(g_qc = 80, g_eff = 15):
    """g_qc = 130, g_qc = 80"""
    """Calculate detuning of coupler, delta=-g_qc**2/(g_eff+5), 5 is direct QQcoupling"""
    delta = - g_qc**2 / (g_eff + 5)
    return delta/1000

def FindPeaks(data, distance=None):
    if distance == None:
        distance = int(1e9/(130*130/0.865)/1e6/2*100000)
    peak_idx, peak_property = find_peaks(data, height=0.05, distance=distance)
    return peak_idx, peak_property

def calcPR(state, type='PR2'):
    """
    Calculate Participation Ratio of given state in product state basis
    type -> PR2 or PRinv
    state is instant
    """
    if type == 'PR2':
        PRi = sum(np.square(np.square(state)))
    elif type == 'PRinv':
        PRi = 1/sum(np.square(state))
    return PRi

def qnumScan(qnumRange=[4,10],gA=15,gB=10):
    qnumList = np.arange(qnumRange[0],qnumRange[1]+1,2)
    tlist = np.arange(0,200,0.01)
    peaks_heights = []
    for qnum in qnumList:
        Fid = ScarRun(qnum, do_plot=0, gA=gA,gB=gB)[0]
        peak_idx, peak_property = FindPeaks(Fid)
        peaks_heights.append(peak_property['peak_heights'][1])

    plt.figure()
    plt.plot(qnumList, peaks_heights, 'o', label='Peaks')
    # plt.plot(tlist[peak_idx[1]], peak_property['peak_heights'][1], 'ro')
    plt.xlabel('qubit numbers', size = 20)
    plt.ylabel("Fidelity", size = 20)
    # plt.title("%d qubits %d couplers"%(qnum,qnum-1), size = 20)
    return peak_property['peak_heights'][1]

# results2,Stlist2, SigmaX2 = EvolveEE(4,3,thermal=1,time=1000)

"""
for dataName in dv.dir(context=ctx)[1]:
    if 'q20_up: IQ raw' in dataName:
    print(dataName)
"""

def fftAmps(freqMax = 0.0334):
    tlist = np.arange(0,200,1.0)
    peakAmps = []
    stateQs = []
    combins = itertools.combinations(range(8), 4)  # int(qnum/2)
    combins = np.asarray([np.asarray(combin) for combin in combins])
    for i in combins:
        results, H, stateQ, popu_t, Imbalance = ScarRun(10,0,0.0,thermal=1, impurity=0.0, idx = i, do_plot=False)
        freq, fourier = fft(tlist, Imbalance, doPlot=False,  des='')
        idx = np.argsort(abs(freq-freqMax))[0]
        peakAmp = fourier[idx]
        peakAmps.append(peakAmp)
        stateQs.append(stateQ)
    plt.figure(434)
    plt.plot(peakAmps,np.arange(len(peakAmps)),'go',markersize=10)
    return peakAmp, stateQs

def fineTune(qnum, cnum = 1, freq = 6.8, gA = 15, gB = 10, modelType = 'comb', idx = [], fig=None,
                        impurity = None, thermal = False, do_plot = True, do_save = False, decoherence = True,
                        do_chop = True, Xtalk = False, improved = True, tlist = None, doEE=False, doScatter=False):
    """
    By changing the impurity and Xtalk, we can tune the theory result to match well with experiment data.
    So it is used along with experimental mat files.
    """
    if tlist is None:
        tlist = np.arange(0,100,1.0)
    if modelType == 'SSH':
        dataScale = scio.loadmat("scarScale.mat")
        plotKey = 'Imbalance' + str(qnum) + 'Q'
        Imbalance = dataScale[plotKey][0]
        delay = dataScale['delay'][0]
    elif modelType == 'comb':
        dataComb = scio.loadmat("Comb_Scar[686]fft.mat")
        if thermal:
            dataComb = scio.loadmat("Comb8Q_Ther[812].mat")
        Imbalance = dataComb['Imbalance'].T[0]
        delay = dataComb['tlist'][0]
    plt.figure(fig)
    plt.plot(delay, Imbalance, 'D-', label='experimental')
    result = ScarRun(qnum, cnum, freq, gA, gB, modelType, idx, fig, impurity, thermal, do_plot, do_save,
                                        decoherence, do_chop, Xtalk, improved, tlist, doEE, doScatter)
    plt.plot(tlist, result[-1], '-', label='theoritical') 
    plt.title(f"{result[2]}",fontsize=20)       
    plt.legend()
    plt.ylim([0.0,1.0])
    plt.tight_layout()
    return

def Scan2D():
    """Scan all the product states basis of 2D lattice to find scar states"""
    combins = itertools.combinations(range(16), 8)
    indexList = np.asarray([combin for combin in combins])
    height_sum = []
    Fidel = {}
    Imbal = {}
    for idx in indexList:
        results, H, stateQ, popu_t, Imbalance, Fid = ScarRun(16,0,freq=0.0,modelType='2D2',idx=idx, do_plot=False)
        peak_idx, peak_property = FindPeaks(Fid, distance=20)
        height_sum.append(sum(peak_property['peak_heights']))
        Fidel.update({stateQ:Fid})
        Imbal.update({stateQ:Imbalance})
    savepath = 'D:/Share/GY/QuantumManybodyScar/20211014Theory2D/'
    saveName1 = 'Fid2D_v2'
    saveName2 = 'Imb2D_v2'
    scio.savemat(savepath+saveName1, Fidel)
    scio.savemat(savepath+saveName2, Imbal)
    
    return indexList, height_sum

def phaseTransition(qnum, cnum=0, modelType='comb', impurity = None, decoherence = False,):
    """
    delta1 is gA(larger); delta2 is gB;
    x-axis : gA/gB -> ratio
    y-axis : gB
    z-axis : <r> & prob(t) & FFT & PR
    """
    RMat = []
    PRMat = []
    height_sum = []
    peak_ti = []
    for ratio in np.linspace(0.1,10,100):
        RCol = []
        PRCol = []
        h_sumCol = []
        peak_idCol = []
        for gB in np.linspace(5,20,16):
            gA = gB * ratio
            tlist = np.arange(0,400,1.0)
            results = ScarRun(qnum=qnum, cnum = cnum, freq = 0.0, gA = gA, gB = gB, 
                            modelType = modelType, idx = [], impurity = impurity, do_plot = False, 
                            decoherence = decoherence, tlist = tlist)
            '''results: result, H, stateQ, popu_t, Fid, P1'''
            distance = 50/gA if 50/gA > 1 else 1
            peak_id,peak_property = find_peaks(results[4], height=0.05, distance=distance)
            h_sumCol.append(peak_property['peak_heights'])
            peak_idCol.append(peak_id)
            PRCol.append(calcPR(results[0][100],type='PR2'))
            RCol.append(results[-1])
        RMat.append(RCol)
        PRMat.append(PRCol)
        height_sum.append(h_sumCol)
        peak_ti.append(peak_idCol)
        savepath = ''
        fileName = f'{qnum}q{cnum}c'
        # savepath = 'DATAnew/' + modelType + f'{qnum}q' + '/'
        fileName += 'Co' if decoherence is True else ''
        fileName += 'Im' if impurity is not None else ''
        scio.savemat(savepath+'PhaseTrans'+fileName+'.mat',
                    {"R":RMat,"PR":PRMat,'height':height_sum, "peak_id":peak_ti})
    return

def plotPhaseTrans(qnum, cnum=0, modelType='comb', impurity=None, decoherence = False):
    """Plot the phase transition of Data from function phaseTransition"""
    fileName = f'{qnum}q{cnum}c'
    savepath = ''
    # savepath = 'DATAnew/' + modelType + f'{qnum}q' + '/'
    fileName += 'Co' if decoherence is True else ''
    fileName += 'Im' if impurity is not None else ''
    data = scio.loadmat(savepath+'PhaseTrans'+fileName+'.mat')
    ratio = np.linspace(0.1,10,100)
    gB = np.linspace(5,20,16)
    RMat = data['R']
    PRMat = data['PR']
    heights = data['height']
    peak_ids = data['peak_id']
    peak_1st = np.zeros((len(ratio),len(gB)))
    for rawi,valueR in enumerate(heights):
        for coli,valueC in enumerate(valueR):
            peak_1st[rawi,coli] = 0 if (len(valueC) == 0) else heights[rawi,coli].flatten()[0]
    htSum = []
    for Raw in heights:
        htSum.append([sum(Raw_i.flatten()) for Raw_i in Raw])
    htSum = np.array(htSum)
    PRs = []
    for Raw in PRMat:
        PRs.append([sum(Raw_i.flatten()) for Raw_i in Raw])
    PRs = np.array(PRs)

    plt.figure()
    plt.subplot(221)
    plt.plot(ratio,peak_1st[:,5],'o')
    plt.xlabel("gA/gB",size=20)
    plt.ylabel("Amps of first peak",size=20)

    plt.subplot(222)
    xx,yy = np.meshgrid(gB, ratio)
    plt.pcolor(xx,yy, peak_1st)
    plt.ylabel("gA / gB", size = 20)
    plt.xlabel("gB", size = 20)
    plt.title("First Peak Amp",size = 15)
    plt.colorbar()

    plt.subplot(223)
    plt.pcolor(xx,yy, RMat)
    plt.ylabel("gA / gB", size = 20)
    plt.xlabel("gB", size = 20)
    plt.title("<R>",size = 15)
    plt.colorbar()

    plt.subplot(224)
    plt.pcolor(xx,yy, PRs)
    plt.ylabel("gA / gB", size = 20)
    plt.xlabel("gB", size = 20)
    plt.title("PR at 100ns",size = 15)
    plt.colorbar()
    
    plt.tight_layout()
    return RMat, PRs, htSum, peak_ids

def plotPRImbalance(qnum, cnum=0, modelType='comb', thermal=False, idx=[]):
    results = ScarRun(qnum=qnum, cnum = cnum, freq = 0.0, modelType = modelType, 
                        thermal=thermal, idx = idx)
    '''results: result, H, stateQ, popu_t, Fid, P1, R'''
    Imbalance = results[4]
    state_t = results[0]
    Basis = np.array(getBasis(qnum,cnum))
    PRt = []
    for state in state_t:
        PRt.append(sum(np.dot(np.square(state.full()).T.flatten(), Basis))/int(qnum/2))
    fig = plt.figure()
    tlist = np.arange(0,200,1.0)
    tlist1 = np.arange(0,400,2.0)
    plt.plot(tlist1,PRt,'bo-')
    plt.plot(tlist, Imbalance, 'ro-')
    # plt.plot(tlist, Imbalance, 'ro-')
    return PRt