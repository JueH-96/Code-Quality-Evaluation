import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    chords = []
    size = 2 * n
    other_end = [0] * (size + 1)
    index = 1
    for _ in range(n):
        a = int(data[index])
        b = int(data[index + 1])
        index += 2
        other_end[a] = b
        other_end[b] = a
    
    stack = []
    for i in range(1, size + 1):
        if other_end[i] > i:
            stack.append(other_end[i])
        else:
            if not stack:
                print("Yes")
                return
            if stack[-1] == i:
                stack.pop()
            else:
                print("Yes")
                return
    print("No")

if __name__ == "__main__":
    main()