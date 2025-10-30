import sys

def main():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    
    counts = [0] * (N + 1)
    max_count = 0
    current_leader = 0
    
    for a in A:
        counts[a] += 1
        if counts[a] > max_count:
            max_count = counts[a]
            current_leader = a
        elif counts[a] == max_count:
            if a < current_leader:
                current_leader = a
        print(current_leader)
        
if __name__ == "__main__":
    main()