import sys
import threading

def main():
    import sys
    input = sys.stdin.readline
    
    N = int(input().strip())
    A = list(map(int, input().split()))
    
    # Count the occurrences of each integer
    freq = {}
    for x in A:
        freq[x] = freq.get(x, 0) + 1
    
    # Find the index with unique value and maximum integer
    best_value = -1
    best_index = -1
    for idx, x in enumerate(A, start=1):
        if freq[x] == 1:
            if x > best_value:
                best_value = x
                best_index = idx
    
    # Output result
    if best_index == -1:
        print(-1)
    else:
        print(best_index)

if __name__ == "__main__":
    main()