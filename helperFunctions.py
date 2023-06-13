import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from bankingSystem import * 
import multiprocessing
from functools import partial


def run(iRun, params):
    # simulation and data collection
    simulationSteps = 500
    model = bankingSystem(params, seed = iRun)
    model.datacollector.collect(model)
    for _ in range(simulationSteps):
        model.simulate()
    agent_data = model.datacollector.get_agent_vars_dataframe()
    model_data = model.datacollector.get_model_vars_dataframe()
    return agent_data, model_data

def multiRun(numberOfruns, params):
    # running the simulation in parallel
    cpuNum = 48
    results = []
    batchNum = 0
    while numberOfruns > 0:
        print(numberOfruns)
        if numberOfruns > cpuNum:
            numberOfruns = numberOfruns - cpuNum
            with multiprocessing.Pool() as pool:
                # run the function in parallel on the input values
                results = results + pool.map(partial(run, params=params), range(batchNum*cpuNum, (batchNum+1)*cpuNum))
            batchNum = batchNum + 1
        else:
            with multiprocessing.Pool() as pool:
                results = results + pool.map(partial(run, params=params), range(batchNum*cpuNum, batchNum*cpuNum + numberOfruns))
            numberOfruns = 0
    return results

# create network graph
def netWorkGraph(matrix, model, printLabel=True):
    size = model.N
    # Create a graph object with 5 nodes
    G = nx.DiGraph(seed=1)
    G.add_nodes_from(list(range(size)))
    # Create a list of edge weights
    weightedEdges = []
    for i in range(size):
        for j in range(size):
            if matrix[i][j] > 0.2:
                # direction of the edge is the direction of the money flow
                weightedEdges.append((j, i, matrix[i][j]))
    G.add_weighted_edges_from(weightedEdges)
    nodeSize = matrix.sum(axis=0) * 100 
    bigLabelIndex = np.where(nodeSize >= np.percentile(nodeSize, 96))[0]
    bigLabel = [model.banks[i] if i in bigLabelIndex else "" for i in range(size)]
    # Set the labels for the nodes using a list of variables
    label_dict = {node: label for node, label in zip(G.nodes, bigLabel)}
    edges = G.edges()
    edgesWidth = [G[u][v]['weight'] * 2 for u,v in edges]
    # change the color of the center nodes
    node_colors = ['red' if node in bigLabelIndex else 'lightblue' for node in G.nodes()]
    pos = nx.fruchterman_reingold_layout(G, scale=10)
    nx.draw_networkx_nodes(G, pos, node_size=nodeSize,node_color=node_colors, alpha=0.9)
    if printLabel:
        label_options = {"ec": "k", "fc": "white", "alpha": 0.7}
        nx.draw_networkx_labels(G, pos,labels = label_dict,
                                font_size=10, bbox=label_options)
    nx.draw_networkx_edges(G, pos, width=edgesWidth, alpha=0.5, connectionstyle="arc3,rad=0.05")
    plt.axis('off')
    plt.show()

def netWorkGraph2(matrix, model, printLabel=True):
    size = model.N
    # Create a graph object with 5 nodes
    G = nx.DiGraph(seed=1)
    G.add_nodes_from(list(range(size)))
    # Create a list of edge weights
    weightedEdges = []
    for i in range(size):
        for j in range(size):
            if matrix[i][j] > 0.02:
                # direction of the edge is the direction of the money flow
                weightedEdges.append((j, i, matrix[i][j]))
    G.add_weighted_edges_from(weightedEdges)
    nodeSize = matrix.sum(axis=0) * 500
    bigLabelIndex = np.where(nodeSize >= np.percentile(nodeSize, 96))[0]
    bigLabel = [model.banks[i] if i in bigLabelIndex else "" for i in range(size)]
    # Set the labels for the nodes using a list of variables
    label_dict = {node: label for node, label in zip(G.nodes, bigLabel)}
    edges = G.edges()
    edgesWidth = [G[u][v]['weight'] * 5 for u,v in edges]
    # change the color of the center nodes
    node_colors = ['red' if node in bigLabelIndex else 'lightblue' for node in G.nodes()]
    pos = nx.fruchterman_reingold_layout(G, scale=10, k=1, iterations=100)
    nx.draw_networkx_nodes(G, pos, node_size=nodeSize,node_color=node_colors, alpha=0.9)
    if printLabel:
        label_options = {"ec": "k", "fc": "white", "alpha": 0.7}
        nx.draw_networkx_labels(G, pos,labels = label_dict,
                                font_size=10, bbox=label_options)
    nx.draw_networkx_edges(G, pos, width=edgesWidth, alpha=0.5, connectionstyle="arc3,rad=0.05")
    plt.axis('off')
    plt.show()

def simulationMonitor(agent_data, model_data, simulationSteps):
    numberOfDefault = [agent_data.xs(i, level="Step")["Default"].sum() for i in range(simulationSteps)]
    averageLeverage = [agent_data.xs(i, level="Step")["Leverage"].sum() / (100 - agent_data.xs(i, level="Step")["Default"].sum()) for i in range(simulationSteps)]
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    fig.set_size_inches(40, 20)
    ax1.set_title("Single simulation average leverage")
    ax1.plot(range(simulationSteps), averageLeverage)
    portfollioValue = [agent_data.xs(i, level="Step")["PortfolioValue"].sum() for i in range(simulationSteps)]
    ax2.plot(range(simulationSteps), portfollioValue)
    ax2.set_title("Single simulation Aggregated Asset Values")
    ax3.bar(range(1, simulationSteps), np.diff(numberOfDefault))
    ax3.set_title("Single simulation Number of default banks")
    sizeOfBorrowing = np.array([[model_data["Liability Matrix"][i].sum() for i in range(simulationSteps)]])
    ax4.plot(np.array(sizeOfBorrowing).mean(axis=0))
    ax4.set_title("Single simulation Size of borrowing")
    
def simulationMonitorCompare(agent_datas, model_datas, simulationSteps):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig.set_size_inches(40, 20)
    ax1.set_title("Mean Leverage Ratio")
    ax2.set_title("Aggregated Asset Values")
    ax3.set_title("Aggregate Size of Borrowing")
    for i, (agent_data, model_data) in enumerate(zip(agent_datas, model_datas)):
        numberOfDefault, averageLeverage, portfollioValue, sizeOfBorrowing = dataCollect(agent_data, model_data, simulationSteps)
        ax1.plot(range(295,305), averageLeverage[295:305], "-o", label="Number of Shocks: " + str(i+1))
        ax1.set_xticks(range(295,305))
        ax2.plot(range(295,305), portfollioValue[295:305], "-o", label="Number of Shocks: " + str(i+1))
        ax2.set_xticks(range(295,305))
        ax3.plot(range(295,305),np.array(sizeOfBorrowing).mean(axis=0)[295:305], "-o", label="Number of Shocks: " + str(i+1))    
        ax3.set_xticks(range(295,305))
    ax1.legend()
    ax2.legend()
    ax3.legend()
        
 
        
def dataCollect(agent_data, model_data, simulationSteps):
    numberOfDefault = [agent_data.xs(i, level="Step")["Default"].sum() for i in range(simulationSteps)]
    averageLeverage = [agent_data.xs(i, level="Step")["Leverage"].sum() / (100 - agent_data.xs(i, level="Step")["Default"].sum()) for i in range(simulationSteps)]
    portfollioValue = [agent_data.xs(i, level="Step")["PortfolioValue"].sum() for i in range(simulationSteps)]
    sizeOfBorrowing = np.array([[model_data["Liability Matrix"][i].sum() for i in range(simulationSteps)]])
    return numberOfDefault, averageLeverage, portfollioValue, sizeOfBorrowing