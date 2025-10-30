import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+M]))
    
    counts = [0] * (N + 1)
    count_map = {}
    heap = list(range(1, N + 1))
    heapq.heapify(heap)
    current_max = 0
    current_leader = 1 if N >= 1 else None
    
    for c in A:
        prev_count = counts[c]
        new_count = prev_count + 1
        
        if prev_count > 0:
            # Remove from previous count
            if prev_count in count_map:
                entry = count_map[prev_count]
                if c in entry['candidates']:
                    entry['candidates'].remove(c)
                    if c == entry['min_candidate']:
                        entry['is_min_valid'] = False
                    if not entry['candidates']:
                        del count_map[prev_count]
        
        # Add to new_count
        if new_count not in count_map:
            count_map[new_count] = {
                'candidates': set(),
                'min_candidate': None,
                'is_min_valid': True
            }
        new_entry = count_map[new_count]
        new_entry['candidates'].add(c)
        if new_entry['min_candidate'] is None or c < new_entry['min_candidate']:
            new_entry['min_candidate'] = c
            new_entry['is_min_valid'] = True
        counts[c] = new_count
        
        # Update current_max
        if new_count > current_max:
            current_max = new_count
        
        # Determine current_leader
        if current_max == 0:
            # Get from heap
            while heap:
                c_min = heapq.heappop(heap)
                if counts[c_min] == 0:
                    heapq.heappush(heap, c_min)
                    current_leader = c_min
                    break
            else:
                current_leader = None
        else:
            entry = count_map.get(current_max, None)
            if entry is None:
                current_leader = None
            else:
                if not entry['is_min_valid']:
                    if entry['candidates']:
                        new_min = min(entry['candidates'])
                        entry['min_candidate'] = new_min
                        entry['is_min_valid'] = True
                    else:
                        entry['min_candidate'] = None
                current_leader = entry['min_candidate']
        
        print(current_leader)

if __name__ == "__main__":
    main()