def main():
    import sys
    data = sys.stdin.read().splitlines()
    n = int(data[0].strip())
    total = 2 * n
    end = [0] * (total + 1)
    
    index = 1
    for _ in range(n):
        a, b = map(int, data[index].split())
        index += 1
        end[a] = b
        end[b] = a
        
    stack = []
    for i in range(1, total + 1):
        if end[i] > i:
            stack.append(end[i])
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