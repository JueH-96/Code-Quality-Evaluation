# YOUR CODE HERE
def find_columns(A):
    total_people = 400
    if total_people % A == 0:
        return total_people // A
    else:
        return -1

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    A = int(input().strip())
    B = find_columns(A)
    print(B)