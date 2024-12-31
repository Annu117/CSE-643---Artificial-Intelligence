import numpy as np
import pickle

def is_connected(adj_matrix, start_node, goal_node):
    n = len(adj_matrix)
    visited = set()
    queue = deque([start_node])

    while queue:
        cur_node = queue.popleft()
        if cur_node == goal_node:
            return True   

        visited.add(cur_node)
        for neighbor, is_connected in enumerate(adj_matrix[cur_node]):
            if is_connected and neighbor not in visited:
                queue.append(neighbor)
    return False 
# General Notes:
# - Update the provided file name (code_<RollNumber>.py) as per the instructions.
# - Do not change the function name, number of parameters or the sequence of parameters.
# - The expected output for each function is a path (list of node names)
# - Ensure that the returned path includes both the start node and the goal node, in the correct order.
# - If no valid path exists between the start and goal nodes, the function should return None.


# Algorithm: Iterative Deepening Search (IDS)

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 2, 9, 8, 5, 97, 98, 12] 

def get_ids_path(adj_matrix, start_node, goal_node):
  if not is_connected(adj_matrix, start_node, goal_node):
        return None 
  def dfs(cur_node, goal_node, depth, path, visited):
      if cur_node == goal_node:
            return path
      if depth == 0:
            return None
      
      visited.add(cur_node)
      for neighbor, is_connected in enumerate(adj_matrix[cur_node]):
          if is_connected and neighbor not in visited:
              result = dfs(neighbor, goal_node, depth - 1, path + [neighbor], visited)
              if result is not None:
                  return result
      visited.remove(cur_node)
      return None

  max_depth = len(adj_matrix)
  for depth in range(1, max_depth + 1):  
      visited = set()
      path = dfs(start_node, goal_node, depth, [start_node], visited)
      if path is not None:
          return path
  return None 

# Algorithm: Bi-Directional Search

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 2, 9, 8, 5, 97, 98, 12]

from collections import deque
def get_bidirectional_search_path(adj_matrix, start_node, goal_node):
    n = len(adj_matrix)  
    
    if start_node == goal_node:
        return [start_node]
    
    #BFS data structures 
    forward_queue = deque([start_node])
    backward_queue = deque([goal_node])
    
    forward_visited = {start_node: None}  # Node -> parent 
    backward_visited = {goal_node: None} 
    
    def explore_neighbors(current_node, queue, visited, other_visited):
        for neighbor, cost in enumerate(adj_matrix[current_node]):
            if cost > 0 and neighbor not in visited:
                queue.append(neighbor)
                visited[neighbor] = current_node
                if neighbor in other_visited:
                    return neighbor
        return None
    while forward_queue and backward_queue:
        # forward BFS
        if forward_queue:
            forward_node = forward_queue.popleft()
            meeting_node = explore_neighbors(forward_node, forward_queue, forward_visited, backward_visited)
            if meeting_node:
                return reconstruct_bidirectional_path(forward_visited, backward_visited, meeting_node)
        
        # backward BFS
        if backward_queue:
            backward_node = backward_queue.popleft()
            meeting_node = explore_neighbors(backward_node, backward_queue, backward_visited, forward_visited)
            if meeting_node:
                return reconstruct_bidirectional_path(forward_visited, backward_visited, meeting_node)

    return None

def reconstruct_bidirectional_path(forward_visited, backward_visited, meeting_node):
    # Reconstructed forward path from start to meeting node
    path = []
    current = meeting_node
    while current is not None:
        path.append(current)
        current = forward_visited[current]
    
    # Reversed the forward path to get from start -> meeting node
    path.reverse()
    
    # Reconstructed backward path from meeting node -> goal
    current = backward_visited[meeting_node]
    while current is not None:
        path.append(current)
        current = backward_visited[current]
    
    return path

# Algorithm: A* Search Algorithm

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - node_attributes: Dictionary of node attributes containing x, y coordinates for heuristic calculations.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 28, 10, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 27, 9, 8, 5, 97, 28, 10, 12]
import heapq
import math
def euclidean_distance(node1, node2, node_attributes):
    x1, y1 = node_attributes[node1]['x'], node_attributes[node1]['y']
    x2, y2 = node_attributes[node2]['x'], node_attributes[node2]['y']
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def get_astar_search_path(adj_matrix, node_attributes, start_node, goal_node):
    n = len(adj_matrix)
    open_set = [(0, start_node)]  # Priority queue (min-heap), stores (f_score, node)
    came_from = {start_node: None}
    g_score = {node: float('inf') for node in range(n)}
    g_score[start_node] = 0
    closed_set = set()
    
    while open_set:
        # Poped the node with the smallest f_score
        _, current = heapq.heappop(open_set)
        if current in closed_set:
            continue
        if current == goal_node:
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            return path[::-1]
        
        closed_set.add(current)
        for neighbor, is_connected in enumerate(adj_matrix[current]):
            if is_connected and neighbor not in closed_set:
                tentative_g_score = g_score[current] + adj_matrix[current][neighbor]
                
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = g_score[neighbor] + euclidean_distance(neighbor, goal_node, node_attributes)
                    heapq.heappush(open_set, (f_score, neighbor))
    
    return None  


# Algorithm: Bi-Directional Heuristic Search

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - node_attributes: Dictionary of node attributes containing x, y coordinates for heuristic calculations.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 34, 33, 11, 32, 31, 3, 5, 97, 28, 10, 12]

def build_path(meeting_node, start_parent_map, goal_parent_map):
    path = []
    # start to meeting node
    node = meeting_node
    while node is not None:
        path.append(node)
        node = start_parent_map.get(node)
    path.reverse()

    # meeting node to goal
    node = goal_parent_map.get(meeting_node)
    while node is not None:
        path.append(node)
        node = goal_parent_map.get(node)
    
    return path

import heapq
from math import inf

def get_bidirectional_heuristic_search_path(adj_matrix, node_attributes, start_node, goal_node):
    
    def reconstruct_path(forward_came_from, backward_came_from, meeting_point):
        path = []
        # path from start -> meeting_point
        cur_node = meeting_point
        while cur_node is not None:
            path.append(cur_node)
            cur_node = forward_came_from.get(cur_node)
        
        path.reverse()

        # Appending path from meeting_point -> goal
        cur_node = backward_came_from.get(meeting_point)
        while cur_node is not None:
            path.append(cur_node)
            cur_node = backward_came_from.get(cur_node)
        
        return path

    # Initiaingl setup for forward and backward searches
    forward_open_set = [(0, start_node)]
    backward_open_set = [(0, goal_node)]
    
    forward_g_score = {start_node: 0}
    backward_g_score = {goal_node: 0}
    
    forward_came_from = {}
    backward_came_from = {}
    
    visited_forward = set()
    visited_backward = set()
    
    meeting_point = None

    while forward_open_set and backward_open_set:
        _, cur_forward = heapq.heappop(forward_open_set)
        visited_forward.add(cur_forward)
        
        _, cur_backward = heapq.heappop(backward_open_set)
        visited_backward.add(cur_backward)
        
        # forward and backward searches meet check
        if cur_forward in visited_backward or cur_backward in visited_forward:
            meeting_point = cur_forward if cur_forward in visited_backward else cur_backward
            break
        
        # Neighbors expansion for forward search
        for neighbor, cost in enumerate(adj_matrix[cur_forward]):
            if cost > 0:
                tentative_g = forward_g_score[cur_forward] + cost
                if neighbor not in forward_g_score or tentative_g < forward_g_score[neighbor]:
                    forward_g_score[neighbor] = tentative_g
                    f_cost = tentative_g + euclidean_distance(neighbor, goal_node, node_attributes)
                    heapq.heappush(forward_open_set, (f_cost, neighbor))
                    forward_came_from[neighbor] = cur_forward
        
        # Neighbors expantion for backward search
        for neighbor, cost in enumerate(adj_matrix[cur_backward]):
            if cost > 0:
                tentative_g = backward_g_score[cur_backward] + cost
                if neighbor not in backward_g_score or tentative_g < backward_g_score[neighbor]:
                    backward_g_score[neighbor] = tentative_g
                    f_cost = tentative_g + euclidean_distance(neighbor, start_node, node_attributes)
                    heapq.heappush(backward_open_set, (f_cost, neighbor))
                    backward_came_from[neighbor] = cur_backward
    
    # reconstructed path output if meeting point is found
    return reconstruct_path(forward_came_from, backward_came_from, meeting_point) if meeting_point else None

# Bonus Problem
 
# Input:
# - adj_matrix: A 2D list or numpy array representing the adjacency matrix of the graph.

# Return:
# - A list of tuples where each tuple (u, v) represents an edge between nodes u and v.
#   These are the vulnerable roads whose removal would disconnect parts of the graph.

# Note:
# - The graph is undirected, so if an edge (u, v) is vulnerable, then (v, u) should not be repeated in the output list.
# - If the input graph has no vulnerable roads, return an empty list [].

# def bonus_problem(adj_matrix):

#   return []
def bonus_problem(adj_matrix):
    n = len(adj_matrix)  
    bridges = []
    
    discovery_time = [-1] * n
    low = [-1] * n
    parent = [-1] * n
    time = [0]

    def dfs(u):
        discovery_time[u] = low[u] = time[0]
        time[0] += 1

        for v in range(n):
            if adj_matrix[u][v] > 0:  # There's an edge between u and v
                if discovery_time[v] == -1:  
                    parent[v] = u
                    dfs(v)
                    low[u] = min(low[u], low[v])

                    # identifying bridges
                    if low[v] > discovery_time[u]:
                        bridges.append((u, v))

                elif v != parent[u]:  # Back edge found, but not the direct parent
                    low[u] = min(low[u], discovery_time[v])

    # Running DFS for all components (disconnected graph)
    for i in range(n):
        if discovery_time[i] == -1:
            dfs(i)

    return bridges

import time
import tracemalloc
from collections import deque
import matplotlib.pyplot as plt

def measure_execution_and_memory(search_func, adj_matrix, node_attributes, start_node, end_node):
    start_time = time.time()

    # Measures memory
    tracemalloc.start()

    # path-finding function
    if search_func in ['get_astar_search_path', 'get_bidirectional_heuristic_search_path']:
        path = globals()[search_func](adj_matrix, node_attributes, start_node, end_node)
    # elif search_func == 'bonus_problem':
    #     path = globals()[search_func](adj_matrix)
    else:
        path = globals()[search_func](adj_matrix, start_node, end_node)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_time = time.time()
    
    execution_time = end_time - start_time
    memory_usage = peak - current  # Peak memory usage

    return execution_time, memory_usage, path

if __name__ == "__main__":
  adj_matrix = np.load('IIIT_Delhi.npy')
  with open('IIIT_Delhi.pkl', 'rb') as f:
    node_attributes = pickle.load(f)

  start_node = int(input("Enter the start node: "))
  end_node = int(input("Enter the end node: "))

  print(f'Iterative Deepening Search Path: {get_ids_path(adj_matrix,start_node,end_node)}')
  print(f'Bidirectional Search Path: {get_bidirectional_search_path(adj_matrix,start_node,end_node)}')
  print(f'A* Path: {get_astar_search_path(adj_matrix,node_attributes,start_node,end_node)}')
  print(f'Bidirectional Heuristic Search Path: {get_bidirectional_heuristic_search_path(adj_matrix,node_attributes,start_node,end_node)}')
  print(f'Bonus Problem: {bonus_problem(adj_matrix)}')
      
  search_algorithms = [
        'get_ids_path',
        'get_bidirectional_search_path',
        'get_astar_search_path',
        'get_bidirectional_heuristic_search_path'
  ]
  execution_times = []
  memory_usages = []
  costs_of_traveling = []
  algorithm_names = []
  for search_func in search_algorithms:
    execution_time, memory_usage, path = measure_execution_and_memory(search_func, adj_matrix, node_attributes, start_node, end_node)
    
    print(f'{search_func.replace("_", " ").title()}:')
    print(f'Path: {path}')
    print(f'Execution Time: {execution_time:.6f} seconds')
    print(f'Memory Usage: {memory_usage / 1024:.2f} KB\n')
#   print('Bonus Problem:')
#   execution_time, memory_usage, path = measure_execution_and_memory('bonus_problem', adj_matrix, None, None, None)
#   print(f'Result: {path}')
#   print(f'Execution Time: {execution_time:.6f} seconds')
#   print(f'Memory Usage: {memory_usage / 1024:.2f} KB')

    execution_times.append(execution_time)
    memory_usages.append(memory_usage / 1024)  # Convert bytes to KB
    costs_of_traveling.append(len(path) if path else float('inf'))  # Use path length or infinity if no path
    algorithm_names.append(search_func.replace("_", " ").title())


  # Time vs. Space scatter plot
  plt.figure(figsize=(10, 6))
  plt.scatter(execution_times, memory_usages, color='blue')
  for i, algo in enumerate(algorithm_names):
      plt.text(execution_times[i], memory_usages[i], algo)
  plt.xlabel('Execution Time (seconds)')
  plt.ylabel('Memory Usage (KB)')
  plt.title('Time Efficiency vs. Space Efficiency')
  plt.grid(True)
  plt.show()

    # Optimality vs. Time scatter plot
  plt.figure(figsize=(10, 6))
  plt.scatter(execution_times, costs_of_traveling, color='green')
  for i, algo in enumerate(algorithm_names):
      plt.text(execution_times[i], costs_of_traveling[i], algo)
  plt.xlabel('Execution Time (seconds)')
  plt.ylabel('Cost of Traveling (Path Length)')
  plt.title('Optimality vs. Time Efficiency')
  plt.grid(True)
  plt.show()

    # Optimality vs. Space scatter plot
  plt.figure(figsize=(10, 6))
  plt.scatter(memory_usages, costs_of_traveling, color='red')
  for i, algo in enumerate(algorithm_names):
      plt.text(memory_usages[i], costs_of_traveling[i], algo)
  plt.xlabel('Memory Usage (KB)')
  plt.ylabel('Cost of Traveling (Path Length)')
  plt.title('Optimality vs. Space Efficiency')
  plt.grid(True)
  plt.show()
