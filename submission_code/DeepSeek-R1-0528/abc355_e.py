import sys

def main():
    data = sys.stdin.readline().split()
    if not data:
        return
    N = int(data[0])
    L = int(data[1])
    R = int(data[2])
    
    total = 0
    l = L
    
    while l <= R:
        current_k = 0
        for exp in range(1, N + 1):
            mask = (1 << exp) - 1
            if l & mask:
                break
            end_segment = l + (1 << exp) - 1
            if end_segment > R:
                break
            current_k = exp
        j = l >> current_k
        print(f"? {current_k} {j}")
        sys.stdout.flush()
        
        response = sys.stdin.readline().strip()
        if response == '':
            break
        t = int(response)
        if t == -1:
            break
            
        total = (total + t) % 100
        l = l + (1 << current_k)
        
    print(f"! {total}")
    sys.stdout.flush()

if __name__ == "__main__":
    main()