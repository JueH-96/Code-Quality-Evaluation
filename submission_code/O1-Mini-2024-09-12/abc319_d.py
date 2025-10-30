# YOUR CODE HERE
import sys

def main():
    import sys
    import sys
    def input():
        return sys.stdin.read()

    data = input().split()
    N = int(data[0])
    M = int(data[1])
    L = list(map(int, data[2:2+N]))
    
    low = max(L)
    high = sum(L) + (N-1)
    
    def possible(W):
        count = 1
        current_width = 0
        for l in L:
            if current_width == 0:
                current_width = l
            else:
                if current_width + 1 + l <= W:
                    current_width += 1 + l
                else:
                    count +=1
                    current_width = l
                    if count > M:
                        return False
        return count <= M
    
    while low < high:
        mid = (low + high) // 2
        if possible(mid):
            high = mid
        else:
            low = mid +1
    print(low)

if __name__ == "__main__":
    main()