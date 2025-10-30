# YOUR CODE HERE
def can_rearrange_to_abc(s):
    return sorted(s) == sorted("ABC")

if __name__ == "__main__":
    import sys
    input_string = sys.stdin.read().strip()
    if can_rearrange_to_abc(input_string):
        print("Yes")
    else:
        print("No")