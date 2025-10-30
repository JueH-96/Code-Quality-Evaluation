def main():
    import sys
    data = sys.stdin.readline().split()
    N = int(data[0])
    L = int(data[1])
    R = int(data[2])
    
    total = 0
    current = L
    while current <= R:
        t = 1
        while True:
            next_t = t * 2
            if next_t > (R - current + 1):
                break
            if current & (next_t - 1):
                break
            t = next_t
        
        i_val = t.bit_length() - 1
        j_val = current // t
        
        print(f"? {i_val} {j_val}", flush=True)
        response = sys.stdin.readline().strip()
        if response == '':
            exit(0)
        T = int(response)
        if T == -1:
            exit(1)
            
        total = (total + T) % 100
        current += t
        
    print(f"! {total}", flush=True)

if __name__ == "__main__":
    main()