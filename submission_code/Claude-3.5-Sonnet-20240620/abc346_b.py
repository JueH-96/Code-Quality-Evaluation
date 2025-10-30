# YOUR CODE HERE
def check_piano_keys(W, B):
    pattern = "wbwbwwbwbwbw"
    target = "w" * W + "b" * B
    
    # Create a string that's long enough to potentially contain the target
    extended_pattern = pattern * ((W + B) // len(pattern) + 2)
    
    return "Yes" if target in extended_pattern else "No"

W, B = map(int, input().split())
print(check_piano_keys(W, B))