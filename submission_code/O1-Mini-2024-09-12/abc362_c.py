# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    L = []
    R = []
    for i in range(N):
        Li = int(data[1 + 2*i])
        Ri = int(data[2 + 2*i])
        L.append(Li)
        R.append(Ri)
    
    X = L.copy()
    current_sum = sum(X)
    needed = -current_sum
    total_possible = sum(r - l for l, r in zip(L, R))
    
    if needed < 0 or needed > total_possible:
        print("No")
        return
    
    for i in range(N):
        if needed == 0:
            break
        increment = min(R[i] - X[i], needed)
        X[i] += increment
        needed -= increment
    
    if needed == 0:
        print("Yes")
        print(' '.join(map(str, X)))
    else:
        print("No")

if __name__ == "__main__":
    main()