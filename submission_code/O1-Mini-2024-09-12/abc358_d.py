# YOUR CODE HERE
import sys

def main():
    import sys
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    M = int(data[idx+1])
    idx +=2
    A = list(map(int, data[idx:idx+N]))
    idx +=N
    B = list(map(int, data[idx:idx+M]))
    A.sort()
    B.sort()
    total_sum = 0
    a_ptr = 0
    for b in B:
        while a_ptr < N and A[a_ptr] < b:
            a_ptr +=1
        if a_ptr == N:
            print(-1)
            return
        total_sum += A[a_ptr]
        a_ptr +=1
    print(total_sum)

if __name__ == "__main__":
    main()