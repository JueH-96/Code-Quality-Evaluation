# YOUR CODE HERE
def rearrange_cards(N, K, A):
    # Take K cards from the bottom and place them on top
    new_order = A[N-K:] + A[:N-K]
    return new_order

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:]))
    
    result = rearrange_cards(N, K, A)
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()