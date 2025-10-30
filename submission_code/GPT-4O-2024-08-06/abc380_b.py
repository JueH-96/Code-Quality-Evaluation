# YOUR CODE HERE
def reconstruct_sequence(S):
    # Split the string by '|', ignoring the first empty split
    parts = S.split('|')[1:-1]
    
    # Calculate the number of '-' in each part
    A = [len(part) for part in parts]
    
    # Print the result as space-separated integers
    print(" ".join(map(str, A)))

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    S = input().strip()
    reconstruct_sequence(S)