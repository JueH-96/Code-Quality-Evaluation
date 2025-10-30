import sys
from math import comb

def main():
    sys.setrecursionlimit(1000000)
    N, T, M = map(int, sys.stdin.readline().split())
    edges = []
    for _ in range(M):
        a, b = map(int, sys.stdin.readline().split())
        edges.append((a, b))
    
    # Function to check if the graph is connected
    def is_connected(edges, n):
        if not edges:
            return True
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        visited = [False] * n
        stack = [0]
        visited[0] = True
        count = 1
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
                    count += 1
        return count == n
    
    # Function to get connected components
    def get_connected_components(edges, n):
        if not edges:
            return []
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        visited = [False] * n
        components = []
        for i in range(n):
            if not visited[i]:
                stack = [i]
                visited[i] = True
                component = []
                while stack:
                    u = stack.pop()
                    component.append(u)
                    for v in adj[u]:
                        if not visited[v]:
                            visited[v] = True
                            stack.append(v)
                # Extract edges within the component
                comp_edges = []
                for u in component:
                    for v in component:
                        if u < v and (u, v) in edges:
                            comp_edges.append((u, v))
                components.append((component, comp_edges))
        return components
    
    # Compute chromatic polynomial for a single connected component
    def compute_chromatic_polynomial(edges, n):
        if not edges:
            return {n: 1}
        if is_connected(edges, n):
            e = edges[0]
            G_minus = edges.copy()
            G_minus.remove(e)
            p_minus = compute_chromatic_polynomial(G_minus, n)
            G_contracted = contract_edge(edges, e, n)
            p_contracted = compute_chromatic_polynomial(G_contracted, n - 1)
            return subtract_polynomials(p_minus, p_contracted)
        else:
            components = get_connected_components(edges, n)
            poly = {0: 1}
            for comp_edges, _ in components:
                comp_poly = compute_chromatic_polynomial(comp_edges, len(comp_edges))
                current = multiply_polynomials(poly, comp_poly)
                poly = current
            return poly
    
    # Contract an edge e = (u, v) in the given graph
    def contract_edge(edges, e, n):
        u, v = e
        new_nodes = set()
        for node in range(n):
            if node != u and node != v:
                new_nodes.add(node)
        new_nodes.add(u)
        new_nodes.add(v)
        new_edges = []
        for edge in edges:
            a, b = edge
            if a == v:
                a = u
            elif a == u:
                a = v
            if b == v:
                b = u
            elif b == u:
                b = v
            if a == b:
                continue
            if a < b:
                new_edges.append((a, b))
        # Remove duplicate edges
        unique_edges = set(new_edges)
        unique_edges = list(unique_edges)
        new_edges = []
        seen = set()
        for e in unique_edges:
            if e not in seen:
                seen.add(e)
                new_edges.append(e)
        return new_edges
    
    # Subtract two polynomials
    def subtract_polynomials(a, b):
        result = {}
        for deg in a:
            result[deg] = result.get(deg, 0) + a[deg]
        for deg in b:
            result[deg] = result.get(deg, 0) - b[deg]
        # Remove zero coefficients
        result = {deg: coeff for deg, coeff in result.items() if coeff != 0}
        return result
    
    # Multiply two polynomials
    def multiply_polynomials(a, b):
        result = {}
        for deg_a, coeff_a in a.items():
            for deg_b, coeff_b in b.items():
                total_deg = deg_a + deg_b
                result[total_deg] = result.get(total_deg, 0) + coeff_a * coeff_b
        return result
    
    # Compute the product of polynomials for all components
    component_polynomials = []
    for comp in get_connected_components(edges, N):
        comp_edges = comp[1]
        n = comp[0]
        cp = compute_chromatic_polynomial(comp_edges, n)
        component_polynomials.append(cp)
    
    # Compute P(m) for each m from 0 to T
    product_polys = [1]
    for cp in component_polynomials:
        product_polys = multiply_polynomials(product_polys, cp)
    
    # Inclusion-exclusion sum
    total = 0
    for m in range(T + 1):
        current_product = 1
        for cp in component_polynomials:
            if m in cp:
                current_product *= cp[m]
            else:
                current_product *= 0
        sign = (-1) ** (T - m)
        c = comb(T, T - m)
        term = sign * c * current_product
        total += term
    print(total)

if __name__ == "__main__":
    main()