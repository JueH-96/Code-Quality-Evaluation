def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    A = int(data[1])
    arrivals = list(map(int, data[2:]))
    
    current_finish_time = 0
    results = []
    
    for t in arrivals:
        # The ticket process starts when the person arrives 
        # or when the ticket booth becomes free, whichever is later.
        start_time = max(t, current_finish_time)
        finish_time = start_time + A
        results.append(str(finish_time))
        current_finish_time = finish_time
        
    sys.stdout.write("
".join(results))
    
if __name__ == '__main__':
    main()