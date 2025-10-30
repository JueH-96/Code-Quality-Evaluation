def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    M = int(data[1])
    S = data[2]
    
    def can_satisfy(x, M, S):
        # x: number of purchased logo T-shirts
        # M: number of plain T-shirts
        # S: schedule string
        
        plain_used = 0  # how many plain T-shirts used since last wash
        logo_used = 0   # how many logo T-shirts used since last wash
        
        for c in S:
            if c == '0':
                # Wash all T-shirts
                plain_used = 0
                logo_used = 0
            elif c == '1':
                # Can wear either plain or logo
                # Prefer plain if available; else use logo
                if plain_used < M:
                    plain_used += 1
                else:
                    if logo_used < x:
                        logo_used += 1
                    else:
                        return False
            else:  # c == '2'
                # Must wear a logo T-shirt
                if logo_used < x:
                    logo_used += 1
                else:
                    return False
        
        return True
    
    # We'll do a simple linear search for the minimum x
    for x in range(N + 1):
        if can_satisfy(x, M, S):
            print(x)
            return

# Remember to call main() at the end!
if __name__ == "__main__":
    main()