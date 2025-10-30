import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
        
    n = int(data[0])
    products = []
    index = 1
    for i in range(n):
        t = int(data[index])
        d = int(data[index+1])
        index += 2
        deadline = t + d
        products.append((t, deadline))
        
    products.sort(key=lambda x: (x[0], x[1]))
    
    last = -1
    count = 0
    for t, deadline in products:
        candidate = last + 1
        if candidate < t:
            candidate = t
        if candidate <= deadline:
            count += 1
            last = candidate
            
    print(count)

if __name__ == "__main__":
    main()