'''
Created on Apr 9, 2017

@author: JonTay
'''
import sys

sys.path.append('./burlap.jar')
import java
from collections import defaultdict
from time import clock
from burlap.behavior.policy import Policy;
from burlap.assignment4 import BasicGridWorld;
from burlap.behavior.singleagent import EpisodeAnalysis;
from burlap.behavior.singleagent.auxiliary import StateReachability;
from burlap.behavior.singleagent.auxiliary.valuefunctionvis import ValueFunctionVisualizerGUI;
from burlap.behavior.singleagent.learning.tdmethods import QLearning;
from burlap.behavior.singleagent.planning.stochastic.policyiteration import PolicyIteration;
from burlap.behavior.singleagent.planning.stochastic.valueiteration import ValueIteration;
from burlap.behavior.valuefunction import ValueFunction;
from burlap.domain.singleagent.gridworld import GridWorldDomain;
from burlap.oomdp.core import Domain;
from burlap.oomdp.core import TerminalFunction;
from burlap.oomdp.core.states import State;
from burlap.oomdp.singleagent import RewardFunction;
from burlap.oomdp.singleagent import SADomain;
from burlap.oomdp.singleagent.environment import SimulatedEnvironment;
from burlap.oomdp.statehashing import HashableStateFactory;
from burlap.oomdp.statehashing import SimpleHashableStateFactory;
from burlap.assignment4.util import MapPrinter;
from burlap.oomdp.core import TerminalFunction;
from burlap.oomdp.core.states import State;
from burlap.oomdp.singleagent import RewardFunction;
from burlap.oomdp.singleagent.explorer import VisualExplorer;
from burlap.oomdp.visualizer import Visualizer;
from burlap.assignment4.util import BasicRewardFunction;
from burlap.assignment4.util import BasicTerminalFunction;
from burlap.assignment4.util import MapPrinter;
from burlap.oomdp.core import TerminalFunction;
from burlap.assignment4.EasyGridWorldLauncher import visualizeInitialGridWorld
from burlap.assignment4.util.AnalysisRunner import calcRewardInEpisode, simpleValueFunctionVis, getAllStates
from burlap.behavior.learningrate import ExponentialDecayLR, SoftTimeInverseDecayLR
import csv
from collections import deque
import pickle


def dumpCSV(discount, nIter, times, rewards, steps, convergence, world, method):
    fname = '{} {} {}.csv'.format(world, method, 'Disc ' + str(discount))
    iters = range(1, nIter + 1)
    assert len(iters) == len(times)
    assert len(iters) == len(rewards)
    assert len(iters) == len(steps)
    assert len(iters) == len(convergence)
    with open('./csv/' + fname, 'wb') as f:
        f.write('iter,time,reward,steps,convergence\n')
        writer = csv.writer(f, delimiter=',')
        writer.writerows(zip(iters, times, rewards, steps, convergence))


def dumpCSVp(discount, nIter, times, rewards, steps, convergence, world, method, policy):
    fname = '{} {} {}.csv'.format(world, method, 'Disc ' + str(discount))
    iters = range(1, nIter + 1)
    assert len(iters) == len(times)
    assert len(iters) == len(rewards)
    assert len(iters) == len(steps)
    assert len(iters) == len(convergence)
    assert len(iters) == len(policy)
    # with open(fname, 'wb') as f:
    with open('./csv/' + fname, 'wb') as f:
        f.write('iter,time,reward,steps,convergence,policy\n')
        writer = csv.writer(f, delimiter=',')
        writer.writerows(zip(iters, times, rewards, steps, convergence, policy))


def runEvals(initialState, plan, rewardL, stepL):
    r = []
    s = []
    for trial in range(evalTrials):
        ea = plan.evaluateBehavior(initialState, rf, tf, 200);
        r.append(calcRewardInEpisode(ea))
        s.append(ea.numTimeSteps())
    rewardL.append(sum(r) / float(len(r)))
    stepL.append(sum(s) / float(len(s)))


def comparePolicies(policy1, policy2):
    assert len(policy1) == len(policy1)
    diffs = 0
    for k in policy1.keys():
        if policy1[k] != policy2[k]:
            diffs += 1
    return diffs


def mapPicture(javaStrArr):
    out = []
    for row in javaStrArr:
        out.append([])
        for element in row:
            out[-1].append(str(element))
    return out


def dumpPolicyMap(javaStrArr, fname):
    pic = mapPicture(javaStrArr)
    with open('./pkl/' + fname, 'wb') as f:
        pickle.dump(pic, f)


if __name__ == '__main__':
    world = 'Easy'

    discounts = [0.99,
                 0.5,
                 ]

    for discount in discounts:
        # NUM_INTERVALS = MAX_ITERATIONS = 100;
        evalTrials = 5;
        # evalTrials =1;
        # userMap = [
        #           [ 0, 0, 0, -1, 0, 0, 0, 0, -5, 0],
        #           [ 0, 1, 0, 1, 1, 1, 1, 1, 1, 0],
        #           [ 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
        #           [ 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
        #           [ 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
        #           [ 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
        #           [ 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
        #           [ 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
        #           [ 0, 1, 0, 1, 1, 1, 1, 1, 1, 0],
        #           [ 0, -3, 0, 0, 0, 0, 0, 0, 0, 0],
        #           ]
        # userMap = [[0, 0, 0, 0, 0],
        #            [0, 1, 1, 1, 0],
        #            [0, 1, 0, 1, 0],
        #            [0, 1, 0, 1, 0],
        #            [0, 0, 0, 0, 0]]
        userMap = [[0, 0, 0, -100, 0],
                   [0, 1, 1, 1, -10],
                   [0, 1, 0, 1, 0],
                   [0, 1, 0, 1, 0],
                   [0, 0, 0, 0, 0]]
        n = len(userMap)
        tmp = java.lang.reflect.Array.newInstance(java.lang.Integer.TYPE, [n, n])
        for i in range(n):
            for j in range(n):
                tmp[i][j] = userMap[i][j]
        userMap = MapPrinter().mapToMatrix(tmp)
        maxX = maxY = n - 1

        gen = BasicGridWorld(userMap, maxX, maxY)
        domain = gen.generateDomain()
        initialState = gen.getExampleState(domain);

        rf = BasicRewardFunction(maxX, maxY, userMap)
        tf = BasicTerminalFunction(maxX, maxY)
        env = SimulatedEnvironment(domain, rf, tf, initialState);
        #    Print the map that is being analyzed
        print "/////{} Grid World Analysis/////\n".format(world)
        MapPrinter().printMap(MapPrinter.matrixToMap(userMap));
        visualizeInitialGridWorld(domain, gen, env)

        hashingFactory = SimpleHashableStateFactory()

        allStates = getAllStates(domain, rf, tf, initialState)

        ################Value Iteration
        print "//{} Value Iteration Analysis//".format(world)
        NUM_INTERVALS = MAX_ITERATIONS = 1000;
        increment = MAX_ITERATIONS / NUM_INTERVALS
        iterations = range(1, MAX_ITERATIONS + 1)
        timing = defaultdict(list)
        rewards = defaultdict(list)
        steps = defaultdict(list)
        convergence = defaultdict(list)
        timing['Value'].append(0)
        for nIter in iterations:
            vi = ValueIteration(domain, rf, tf, discount, hashingFactory, -1, nIter);
            # ValueIteration(Domain domain, RewardFunction rf, TerminalFunction tf, double gamma,
            # HashableStateFactory hashingFactory, double maxDelta, int maxIterations)
            vi.setDebugCode(0)
            vi.performReachabilityFrom(initialState)
            vi.toggleUseCachedTransitionDynamics(False)
            startTime = clock()
            vi.runVI()
            timing['Value'].append(timing['Value'][-1] + clock() - startTime)
            p = vi.planFromState(initialState);
            convergence['Value'].append(vi.latestDelta)
            print("convergence delta = " + str(vi.latestDelta))
            current_policy = {state: p.getAction(state).toString() for state in allStates}
            if nIter == 1:
                convergence['Policy'].append(18)
            else:
                convergence['Policy'].append(comparePolicies(last_policy, current_policy))
                print ('convergence policy = ' + str(comparePolicies(last_policy, current_policy)))
            last_policy = current_policy
            # evaluate the policy with evalTrials roll outs
            runEvals(initialState, p, rewards['Value'], steps['Value'])

            if nIter == 1:
                simpleValueFunctionVis(vi, p, initialState, domain, hashingFactory,
                                       "Value Iter {} Disc {}".format(nIter, discount))
                dumpPolicyMap(MapPrinter.printPolicyMap(allStates, p, gen.getMap()),
                              'Value {} Iter {} Disc {} Policy Map.pkl'.format(world, nIter, str(discount)))
            if nIter % 2 == 1:
                simpleValueFunctionVis(vi, p, initialState, domain, hashingFactory,
                                       "Value Iter {} Disc {}".format(nIter, discount))
                dumpPolicyMap(MapPrinter.printPolicyMap(allStates, p, gen.getMap()),
                              'Value {} Iter {} Disc {} Policy Map.pkl'.format(world, nIter, str(discount)))
            if nIter == 5 or vi.latestDelta < 1e-6:
                dumpPolicyMap(MapPrinter.printPolicyMap(allStates, p, gen.getMap()),
                              'Value {} Iter {} Disc {} Policy Map.pkl'.format(world, nIter, str(discount)))
            if vi.latestDelta < 1e-6:
                break
        simpleValueFunctionVis(vi, p, initialState, domain, hashingFactory,
                               "Value Iter {} Disc {}".format(nIter, discount))
        dumpPolicyMap(MapPrinter.printPolicyMap(allStates, p, gen.getMap()),
                      'Value {} Iter {} Disc {} Policy Map.pkl'.format(world, nIter, str(discount)))

        print "\n\n\n"
        # dumpCSV(discount,nIter, timing['Value'][1:], rewards['Value'], steps['Value'],convergence['Value'], world, 'Value')
        dumpCSVp(discount, nIter, timing['Value'][1:], rewards['Value'], steps['Value'], convergence['Value'], world,
                 'Value', convergence['Policy'])

        ################Policy Iteration
        print "//{} Policy Iteration Analysis//".format(world)
        NUM_INTERVALS = MAX_ITERATIONS = 1000;
        # increment = MAX_ITERATIONS/NUM_INTERVALS
        iterations = range(1, MAX_ITERATIONS + 1)
        timing = defaultdict(list)
        rewards = defaultdict(list)
        steps = defaultdict(list)
        convergence = defaultdict(list)
        timing['Policy'].append(0)

        for nIter in iterations:
            pi = PolicyIteration(domain, rf, tf, discount, hashingFactory, 1e-3, 10, nIter)
            # PolicyIteration(Domain domain, RewardFunction rf, TerminalFunction tf, double gamma,
            # HashableStateFactory hashingFactory,
            # double maxDelta, int maxEvaluationIterations, int maxPolicyIterations)
            pi.toggleUseCachedTransitionDynamics(False)
            startTime = clock()
            p = pi.planFromState(initialState);
            timing['Policy'].append(timing['Policy'][-1] + clock() - startTime)
            policy = pi.getComputedPolicy()
            convergence['Delta'].append(pi.lastPIDelta)
            current_policy = {state: policy.getAction(state).toString() for state in allStates}
            print("convergence delta = " + str(pi.lastPIDelta))
            if nIter == 1:
                convergence['Policy'].append(18)
                dumpPolicyMap(MapPrinter.printPolicyMap(allStates, p, gen.getMap()),
                              'Policy {} Iter {} Disc {} Policy Map.pkl'.format(world, nIter, str(discount)))
            else:
                convergence['Policy'].append(comparePolicies(last_policy, current_policy))
                print ('convergence policy = ' + str(comparePolicies(last_policy, current_policy)))
            last_policy = current_policy
            runEvals(initialState, p, rewards['Policy'], steps['Policy'])

            if nIter == 1:
                simpleValueFunctionVis(pi, p, initialState, domain, hashingFactory,
                                       "Policy Iter {} Disc {}".format(nIter, discount))
                dumpPolicyMap(MapPrinter.printPolicyMap(allStates, p, gen.getMap()),
                              'Policy {} Iter {} Disc {} Policy Map.pkl'.format(world, nIter, str(discount)))
            if nIter % 2 == 1:
                simpleValueFunctionVis(pi, p, initialState, domain, hashingFactory,
                                       "Policy Iter {} Disc {}".format(nIter, discount))
                dumpPolicyMap(MapPrinter.printPolicyMap(allStates, p, gen.getMap()),
                              'Policy {} Iter {} Disc {} Policy Map.pkl'.format(world, nIter, str(discount)))
            if nIter == 5 or convergence['Delta'][-1] < 1e-3:
                simpleValueFunctionVis(pi, p, initialState, domain, hashingFactory,
                                       "Policy Iter {} Disc {}".format(nIter, discount))
                dumpPolicyMap(MapPrinter.printPolicyMap(allStates, p, gen.getMap()),
                              'Policy {} Iter {} Disc {} Policy Map.pkl'.format(world, nIter, str(discount)))
            if convergence['Delta'][-1] < 1e-3:
                break

        simpleValueFunctionVis(pi, p, initialState, domain, hashingFactory,
                               "Policy Iter {} Disc {}".format(nIter, discount))
        dumpPolicyMap(MapPrinter.printPolicyMap(allStates, p, gen.getMap()),
                      'Policy {} Iter {} Disc {} Policy Map.pkl'.format(world, nIter, str(discount)))

        MapPrinter.printPolicyMap(pi.getAllStates(), p, gen.getMap());
        print "\n\n\n"
        # dumpCSV(discount,nIter, timing['Policy'][1:], rewards['Policy'], steps['Policy'],convergence['Delta'], world, 'Policy')
        dumpCSVp(discount, nIter, timing['Policy'][1:], rewards['Policy'], steps['Policy'], convergence['Delta'], world,
                 'Policy', convergence['Policy'])

        ################QLearn
        MAX_ITERATIONS = NUM_INTERVALS = 1000;
        MAX_EPISODESIZE = 400
        increment = MAX_ITERATIONS / NUM_INTERVALS
        iterations = range(1, MAX_ITERATIONS + 1)
        show = 0
        for lr in [
            0.1,
            0.9,
        ]:
            for qInit in [
                -100,
                0,
                100,
            ]:
                # for qInit in [1,]:
                for epsilon in [
                    0.1,
                    0.3,
                    0.5,
                ]:
                    print (show)
                    show += 1
                    timing = defaultdict(list)
                    rewards = defaultdict(list)
                    steps = defaultdict(list)
                    convergence = defaultdict(list)
                    # last10Chg = deque([99]*10,maxlen=10)
                    last10Chg = deque([10] * 10, maxlen=10)
                    Qname = 'Q-Learning L{:0.2f} q{:0.1f} E{:0.2f}'.format(lr, qInit, epsilon)
                    # agent = QLearning(domain, discount, hashingFactory, qInit, lr, epsilon, MAX_EPISODESIZE)
                    agent = QLearning(domain, discount, hashingFactory, qInit, lr, epsilon)

                    # QLearning(Domain domain, double gamma, HashableStateFactory hashingFactory,
                    # 			double qInit, double learningRate, double epsilon, int maxEpisodeSize)
                    # agent.setLearningRateFunction(SoftTimeInverseDecayLR(lr,0.))
                    agent.setDebugCode(0)

                    print "//{} {} Iteration Analysis//".format(world, Qname)
                    for nIter in iterations:
                        if nIter % 50 == 0: print(nIter)
                        # agent = QLearning(domain, discount, hashingFactory, qInit, lr, epsilon, 300)

                        print("start learning")
                        startTime = clock()
                        # ea = agent.runLearningEpisode(env, MAX_EPISODESIZE)
                        ea = agent.runLearningEpisode(env)
                        # runLearningEpisode(Environment env, int maxSteps)
                        print("stop learning")
                        env.resetEnvironment()

                        agent.initializeForPlanning(rf, tf, 1)
                        # public void initializeForPlanning(RewardFunction rf, TerminalFunction tf, int numEpisodesForPlanning)
                        p = agent.planFromState(initialState)  # run planning from our initial state
                        if len(timing[Qname]) > 0:
                            timing[Qname].append(timing[Qname][-1] + clock() - startTime)
                        else:
                            timing[Qname].append(clock() - startTime)
                        last10Chg.append(agent.maxQChangeInLastEpisode)
                        convergence[Qname].append(sum(last10Chg) / 10.)
                        # evaluate the policy with one roll out visualize the trajectory
                        runEvals(initialState, p, rewards[Qname], steps[Qname])
                        current_policy = {state: p.getAction(state).toString() for state in allStates}
                        if nIter == 1:
                            convergence['Policy'].append(18)
                        else:
                            convergence['Policy'].append(comparePolicies(last_policy, current_policy))
                            print ('convergence policy = ' + str(comparePolicies(last_policy, current_policy)))
                        last_policy = current_policy

                        if nIter == 1:
                            simpleValueFunctionVis(agent, p, initialState, domain, hashingFactory,
                                                   Qname + ' nIter' + str(nIter) + 'Disc' + str(discount))
                            dumpPolicyMap(MapPrinter.printPolicyMap(allStates, p, gen.getMap()),
                                          'QL {} {} Iter {} Disc {} Policy Map.pkl'.format(world, Qname, nIter, str(discount)))
                        if nIter % 50 == 1:
                            # simpleValueFunctionVis(agent, p, initialState, domain, hashingFactory, Qname+' nIter'+str(nIter) + 'Disc'+str(discount))
                            dumpPolicyMap(MapPrinter.printPolicyMap(allStates, p, gen.getMap()),
                                          'QL {} {} Iter {} Disc {} Policy Map.pkl'.format(world, Qname, nIter, str(discount)))
                        if nIter == 50 or convergence[Qname][-1] < 1e-5:
                            dumpPolicyMap(MapPrinter.printPolicyMap(allStates, p, gen.getMap()),
                                          'QL {} {} Iter {} Disc {} Policy Map.pkl'.format(world, Qname, nIter, str(discount)))
                        if convergence[Qname][-1] < 1e-5:
                            break
                    simpleValueFunctionVis(agent, p, initialState, domain, hashingFactory,
                                           Qname + ' nIter' + str(nIter) + 'Disc' + str(discount))
                    dumpPolicyMap(MapPrinter.printPolicyMap(allStates, p, gen.getMap()),
                                  'QL {} {} Iter {} Disc {} Policy Map.pkl'.format(world, Qname, nIter, str(discount)))
                    print "\n\n\n"
                    # dumpCSV(discount,nIter, timing[Qname], rewards[Qname], steps[Qname],convergence[Qname], world, Qname)
                    dumpCSVp(discount, nIter, timing[Qname], rewards[Qname], steps[Qname], convergence[Qname], world,
                             Qname, convergence['Policy'])

                    print "done with #" + str(show)
print "============================================================================================================================all done"
