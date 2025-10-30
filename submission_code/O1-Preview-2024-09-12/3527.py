class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        n = len(colors)
        from collections import defaultdict

        # Build initial runs
        # Each run is represented as (start index, length)
        runs = []
        idx_to_run = {}
        counts = defaultdict(int)  # Counts of runs per length

        i = 0
        while i < n:
            start = i
            length = 1
            while True:
                next_i = (i + 1) % n
                if next_i == start:
                    break  # Completed the circle
                if colors[i] != colors[next_i]:
                    length += 1
                    i = next_i
                    if i == start:
                        break
                else:
                    break
            runs.append({'start': start, 'length': length})
            for j in range(length):
                idx = (start + j) % n
                idx_to_run[idx] = runs[-1]
            counts[length] += 1
            i = (start + length) % n
            if i == 0:
                break  # Completed the circle

        answer = []

        for query in queries:
            if query[0] == 1:
                # Type 1 query: Count of alternating groups with size size_i.
                size_i = query[1]
                total = 0
                for length in counts:
                    if length >= size_i:
                        total += counts[length] * (length - size_i + 1)
                answer.append(total)
            else:
                # Type 2 query: Change colors[index_i] to color_i.
                index_i, color_i = query[1], query[2]
                if colors[index_i] == color_i:
                    continue  # No change
                # Update colors[index_i]
                colors[index_i] = color_i
                # Fetch runs before and after index_i
                prev_idx = (index_i - 1 + n) % n
                next_idx = (index_i + 1) % n
                run_i = idx_to_run[index_i]
                run_prev = idx_to_run.get(prev_idx, None)
                run_next = idx_to_run.get(next_idx, None)

                # Remove old runs from counts
                counts[run_i['length']] -= 1
                if counts[run_i['length']] == 0:
                    del counts[run_i['length']]

                # Determine if we need to merge or split runs
                need_merge_prev = colors[prev_idx] != colors[index_i]
                need_merge_next = colors[index_i] != colors[next_idx]
                new_runs = []

                # Remove runs from counts and idx_to_run
                runs_to_remove = [run_i]
                if need_merge_prev and run_prev and run_prev != run_i:
                    counts[run_prev['length']] -= 1
                    if counts[run_prev['length']] == 0:
                        del counts[run_prev['length']]
                    runs_to_remove.append(run_prev)
                if need_merge_next and run_next and run_next != run_i:
                    counts[run_next['length']] -= 1
                    if counts[run_next['length']] == 0:
                        del counts[run_next['length']]
                    runs_to_remove.append(run_next)

                # Remove old runs
                for run in runs_to_remove:
                    for j in range(run['length']):
                        idx = (run['start'] + j) % n
                        del idx_to_run[idx]
                    if run in runs:
                        runs.remove(run)

                # Build new runs
                if need_merge_prev and need_merge_next:
                    # Merge prev, curr, next into one run
                    new_start = run_prev['start'] if run_prev else index_i
                    new_length = run_prev['length'] + run_i['length'] + run_next['length'] if run_prev and run_next else (run_prev['length'] if run_prev else 0) + run_i['length'] + (run_next['length'] if run_next else 0)
                    new_run = {'start': new_start, 'length': new_length}
                    runs.append(new_run)
                    counts[new_length] +=1
                    for j in range(new_length):
                        idx = (new_start + j) % n
                        idx_to_run[idx] = new_run
                elif need_merge_prev:
                    # Merge prev and curr
                    new_start = run_prev['start'] if run_prev else index_i
                    new_length = run_prev['length'] + run_i['length'] if run_prev else run_i['length']
                    new_run = {'start': new_start, 'length': new_length}
                    runs.append(new_run)
                    counts[new_length] += 1
                    for j in range(new_length):
                        idx = (new_start + j) % n
                        idx_to_run[idx] = new_run
                elif need_merge_next:
                    # Merge curr and next
                    new_start = index_i
                    new_length = run_i['length'] + run_next['length'] if run_next else run_i['length']
                    new_run = {'start': new_start, 'length': new_length}
                    runs.append(new_run)
                    counts[new_length] += 1
                    for j in range(new_length):
                        idx = (new_start + j) % n
                        idx_to_run[idx] = new_run
                else:
                    # No merge, create new run at index_i
                    new_run = {'start': index_i, 'length': run_i['length']}
                    runs.append(new_run)
                    counts[run_i['length']] += 1
                    idx_to_run[index_i] = new_run
        return answer