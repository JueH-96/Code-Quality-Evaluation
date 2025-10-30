from sortedcontainers import SortedList

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    Q = int(data[2])
    
    A = [0] * N
    Sorted_A = SortedList(A)
    
    Top_K_sum = sum(Sorted_A[-K:])
    
    index = 3
    for _ in range(Q):
        X_i = int(data[index]) - 1  # Convert to 0-based index
        Y_i = int(data[index + 1])
        index += 2
        
        old_value = A[X_i]
        Sorted_A.remove(old_value)
        A[X_i] = Y_i
        Sorted_A.add(Y_i)
        
        if Y_i >= Sorted_A[-K]:
            if old_value < Sorted_A[-K]:
                Top_K_sum += Y_i - Sorted_A[-K]
            else:
                Top_K_sum += Y_i - old_value
        else:
            if old_value > Sorted_A[-K]:
                Top_K_sum += Sorted_A[-K] - old_value
        
        print(Top_K_sum)

if __name__ == "__main__":
    main()