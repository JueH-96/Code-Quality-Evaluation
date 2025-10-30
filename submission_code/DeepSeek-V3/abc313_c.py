# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    if N == 1:
        print(0)
        return
    
    total = sum(A)
    avg = total // N
    remainder = total % N
    
    target = [avg] * (N - remainder) + [avg + 1] * remainder
    target.sort()
    A.sort()
    
    operations = 0
    for a, t in zip(A, target):
        operations += abs(a - t)
    
    print(operations // 2)

if __name__ == "__main__":
    main()