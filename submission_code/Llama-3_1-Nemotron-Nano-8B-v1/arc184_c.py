def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    events_even = []
    events_odd = []
    
    for d in range(1, 101):
        L_d = (1 << (d-1)) - 1
        if d == 1:
            terms = 1
        else:
            terms = 1 << (d-2)
        
        for a_k in A:
            a = L_d - a_k
            if a % 2 == 0:
                parity = 'even'
            else:
                parity = 'odd'
            
            if parity == 'even':
                start = a
                if d == 1:
                    end = start
                else:
                    end = start + 2 * (terms - 1)
                if start > end:
                    continue
                effective_start = start
                if effective_start < 0:
                    continue
                events_even.append((effective_start, 1))
                events_even.append((end + 1, -1))
            else:
                start = a
                if d == 1:
                    end = start
                else:
                    end = start + 2 * (terms - 1)
                if start > end:
                    continue
                effective_start = start
                if effective_start < 0:
                    continue
                events_odd.append((effective_start, 1))
                events_odd.append((end + 1, -1))
    
    def compute_max(events):
        events.sort()
        max_count = 0
        current = 0
        for pos, delta in events:
            current += delta
            if current > max_count:
                max_count = current
        return max_count
    
    max_even = compute_max(events_even)
    max_odd = compute_max(events_odd)
    
    print(max(max_even, max_odd))

if __name__ == '__main__':
    main()