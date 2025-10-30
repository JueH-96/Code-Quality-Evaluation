import sys

def main():
    input = sys.stdin.readline
    N = int(input().strip())
    A = list(map(int, input().split()))
    
    # count how many times we've seen each number so far,
    # and record the index of the 2nd occurrence (middle).
    count = [0] * (N + 1)
    middle_index = [0] * (N + 1)
    
    for idx, x in enumerate(A, start=1):
        count[x] += 1
        if count[x] == 2:
            middle_index[x] = idx
    
    # sort 1..N by their middle occurrence index
    result = list(range(1, N + 1))
    result.sort(key=lambda i: middle_index[i])
    
    # output
    print(*result)

if __name__ == "__main__":
    main()