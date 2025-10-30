class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        n = 0
        for edge in edges:
            n = max(n, edge[0], edge[1])
        n += 1
        edge_list = []
        for i in range(len(edges)):
            u, v, w = edges[i]
            edge_list.append({'u': u, 'v': v, 'w': w, 'index': i, 'removed': False})
        
        for node in range(n):
            while True:
                incident_edges = []
                degree = 0
                for edge_obj in edge_list:
                    if not edge_obj['removed'] and (edge_obj['u'] == node or edge_obj['v'] == node):
                        degree += 1
                        incident_edges.append(edge_obj)
                
                if degree <= k:
                    break
                    
                incident_edges.sort(key=lambda e: e['w'])
                
                removed_count = 0
                for edge_obj in incident_edges:
                    if not edge_obj['removed'] and (edge_obj['u'] == node or edge_obj['v'] == node):
                        edge_obj['removed'] = True
                        removed_count += 1
                        if removed_count == degree - k:
                            break
                            
        max_weight_sum = 0
        for edge_obj in edge_list:
            if not edge_obj['removed']:
                max_weight_sum += edge_obj['w']
                
        return max_weight_sum