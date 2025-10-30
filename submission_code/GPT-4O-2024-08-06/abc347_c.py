# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = int(data[1])
    B = int(data[2])
    D = list(map(int, data[3:]))
    
    week_length = A + B
    
    for d in D:
        if not (1 <= (d % week_length) <= A):
            print("No")
            return
    
    print("Yes")