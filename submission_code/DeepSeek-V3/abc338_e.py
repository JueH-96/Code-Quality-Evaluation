# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    chords = []
    index = 1
    for _ in range(N):
        a = int(data[index])
        b = int(data[index+1])
        if a > b:
            a, b = b, a
        chords.append((a, b))
        index += 2
    
    # Sort chords based on the first endpoint
    chords.sort()
    
    # Check for intersections
    stack = []
    for chord in chords:
        a, b = chord
        while stack and stack[-1][1] < a:
            stack.pop()
        if stack and stack[-1][1] > a and stack[-1][1] < b:
            print("Yes")
            return
        stack.append(chord)
    
    print("No")

if __name__ == "__main__":
    main()