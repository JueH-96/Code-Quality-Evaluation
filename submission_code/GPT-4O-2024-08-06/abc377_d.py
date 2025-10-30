def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    intervals = []
    index = 2
    for _ in range(N):
        L = int(data[index])
        R = int(data[index + 1])
        intervals.append((L, R))
        index += 2
    
    # Total number of pairs (l, r) with 1 <= l <= r <= M
    total_pairs = M * (M + 1) // 2
    
    # Subtract the forbidden pairs
    forbidden_pairs = 0
    for L, R in intervals:
        forbidden_pairs += L * (M - R + 1)
    
    # Result is total pairs minus forbidden pairs
    result = total_pairs - forbidden_pairs
    print(result)