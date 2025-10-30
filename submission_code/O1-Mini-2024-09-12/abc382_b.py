# YOUR CODE HERE
def main():
    N, D = map(int, input().split())
    S = list(input())
    
    # Find all indices with '@'
    at_indices = [i for i, c in enumerate(S) if c == '@']
    
    # Take the last D indices and set them to '.'
    for i in at_indices[-D:]:
        S[i] = '.'
    
    # Convert back to string and print
    print(''.join(S))

if __name__ == "__main__":
    main()