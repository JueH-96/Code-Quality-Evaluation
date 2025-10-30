def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    B = list(map(int, data[N+1:2*N]))
    
    A_sorted = sorted(A, reverse=True)
    B_sorted = sorted(B, reverse=True)
    
    j = 0
    unassigned = []
    for toy in A_sorted:
        while j < len(B_sorted):
            if B_sorted[j] >= toy:
                j += 1
                break
            else:
                j += 1
        else:
            unassigned.append(toy)
    
    if len(unassigned) > 1:
        print(-1)
    elif len(unassigned) == 1:
        print(unassigned[0])
    else:
        print(min(A))

if __name__ == '__main__':
    main()