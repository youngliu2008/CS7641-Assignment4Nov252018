import sys
import os
import time
from array import array
from time import clock
from itertools import product
import matplotlib.pyplot as plt
# from ggplot import *
import csv
import pandas as pd
import numpy as np

# subdir=''
subdir='0.8/'
# subdir='0.8discount/'
subdir='1/'
def plot(worlds, methods, discounts, x, y, z=None):
    plt.clf()
    fig, ax = plt.subplots()
    # imgfile = '@NAME@'+'_@ALG@' + ' dimred' + ".png"
    imgfile = './'+subdir+'img/{} {} {}-{}.png'.format(str(worlds), str(methods),
                                             'Disc ' + str(discounts), x + y)
    for world in worlds:
        for method in methods:
            for discount in discounts:
                fname = './'+subdir+'csv/{} {} {}.csv'.format(world, method,
                                                    'Disc ' + str(discount))
                data = pd.read_csv(fname)
                ax.plot(data[x], data[y],
                        label=method + ' Disc ' + str(discount))
                if z != None:
                    ax.plot(data[x], data[z],
                            label=method + ' Disc ' + str(discount))
    # data = data.loc[data['param_NN__hidden_layer_sizes'] == str(nn_structure)]
    # data = data.loc[data['param_NN__alpha'] == nn_alpha]

    plt.xlabel(x)
    # plt.ylabel(y)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(y + " vs " + x)
    # plt.show()
    plt.savefig(imgfile)

    return None

def plotQL(worlds, lrs, qInits, epsilons, discounts, x, y, z=None):
    plt.clf()
    fig, ax = plt.subplots()
    # imgfile = '@NAME@'+'_@ALG@' + ' dimred' + ".png"

    for world in worlds:
        imgfile = './'+subdir+'img/{} {} {}-{}.png'.format(str(worlds),
                                                 str(lrs) + str(qInits)+ str(epsilons),
                                                 'Disc ' + str(discounts),
                                                 x + y)
        for discount in discounts:
            for lr in lrs:
                for qInit in qInits:
                    for epsilon in epsilons:
                        Qname = 'Q-Learning L{:0.2f} q{:0.1f} E{:0.2f}'.\
                            format(lr,qInit,epsilon)
                        fname = './'+subdir+'csv/{} {} {}.csv'.format(world, Qname,
                                                'Disc ' + str(discount))
                        data = pd.read_csv(fname)
                        ax.plot(data[x], data[y],
                                label='l'+str(lr)+' q'+str(qInit)+
                                      ' e'+str(epsilon)+' d' + str(discount))
                        if z != None:
                            ax.plot(data[x], data[z],
                                    label='l'+str(lr) + ' q' + str(qInit) +
                                          ' e'+str(epsilon) + ' d' + str(discount))

        plt.xlabel(x)
        # plt.ylabel(y)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 1, box.height])
        # ax.legend(loc='best', bbox_to_anchor=(1, 0.5))
        ax.legend(loc='best')
        plt.title(y + " vs " + x)
        # plt.show()
        plt.savefig(imgfile)

    return None

if __name__ == '__main__':
    worlds = [
        'Easy'
    ]
    methods = [
        'Value',#####
        'Policy'######
    ]

    lrs = [
        0.1,####
        # 0.9,
    ]
    qInits = [
            -100,####
            # 0,
            # 100,
    ]
    epsilons = [
                0.1,#####
                # 0.3,
                # 0.5,
    ]

    discounts = [
        # 0.5,
        0.99###
    ]
    xs=['iter','time']
    ys=['convergence','policy','time','reward','steps']
    for x in xs:
        for y in ys:
            plot(worlds=worlds, methods=methods, discounts=discounts, x=x,
                 y=y)
            plotQL(worlds=worlds, lrs = lrs, qInits=qInits, epsilons=epsilons,
                   discounts=discounts, x=x, y=y)

    ##############Hard############
    worlds = [
        'Hard'
    ]
    methods = [
        'Value',
        'Policy'
    ]

    lrs = [
        0.1,####*
        # 0.9,
    ]
    qInits = [
            # -100,
            # 0,
            100,####*
    ]
    epsilons = [
                0.1,####*
                # 0.3,
                # 0.5,
    ]

    discounts = [
        # 0.5,
        0.99
    ]
    xs = ['iter', 'time']
    ys = ['convergence', 'policy', 'time', 'reward', 'steps']
    for x in xs:
        for y in ys:
            plot(worlds=worlds, methods=methods, discounts=discounts, x=x,
                 y=y)
            plotQL(worlds=worlds, lrs=lrs, qInits=qInits, epsilons=epsilons,
                   discounts=discounts, x=x, y=y)