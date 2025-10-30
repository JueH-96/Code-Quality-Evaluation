from collections import defaultdict
import sys

def maximize_minimum_deliciousness():
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    deliciousness_by_color = defaultdict(list)
    
    for i in range(1, N + 1):
        A, C = map(int, data[i].split())
        deliciousness_by_color[C].append(A)
    
    max_min_deliciousness = 0
    
    for deliciousness in deliciousness_by_color.values():
        max_min_deliciousness = max(max_min_deliciousness, min(deliciousness))
    
    print(max_min_deliciousness)

maximize_minimum_deliciousness()