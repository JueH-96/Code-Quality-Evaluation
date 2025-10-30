import sys

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    T = int(data[idx])
    idx += 1
    results = []
    for _ in range(T):
        N = int(data[idx])
        idx += 1
        P = list(map(int, data[idx:idx+N]))
        idx += N
        
        # Compute the maximum m
        m = 0
        while m < N and P[m] == m + 1:
            m += 1
        
        # Count correct elements beyond m
        correct_beyond = 0
        for i in range(m, N):
            if P[i] == i + 1:
                correct_beyond += 1
        
        # Compute the number of gaps
        gaps = 0
        i = m
        while i < N:
            if P[i] == i + 1:
                i += 1
            else:
                gaps += 1
                i += 1
        
        # The minimal number of operations is the number of gaps
        results.append(gaps)
    
    sys.stdout.write("
".join(map(str, results)) + "
")

if __name__ == '__main__':
    main()