import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    if N == 1:
        print(1)
        return
    
    from collections import Counter
    counts = Counter(A)
    
    unique_values = [i for i in counts if counts[i] == 1]
    
    if not unique_values:
        print(-1)
        return
    
    max_value = max(unique_values)
    # Find the index of the person with the max_value
    for i in range(N):
        if A[i] == max_value:
            print(i + 1)
            return