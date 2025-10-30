# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    stack = []
    
    for a in A:
        size = 2 ** a
        stack.append(size)
        
        # Check for possible merges
        while len(stack) > 1 and stack[-1] == stack[-2]:
            # Merge the last two balls
            last = stack.pop()
            stack[-1] = last * 2
    
    print(len(stack))