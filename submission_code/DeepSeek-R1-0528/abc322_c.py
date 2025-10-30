import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    m = int(data[1])
    a_list = list(map(int, data[2:2+m]))
    
    firework = [False] * n
    for a in a_list:
        firework[a-1] = True
        
    res = [0] * n
    for i in range(n-2, -1, -1):
        if not firework[i]:
            res[i] = res[i+1] + 1
            
    for num in res:
        print(num)

if __name__ == "__main__":
    main()