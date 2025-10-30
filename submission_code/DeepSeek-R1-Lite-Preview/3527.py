from typing import List

class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        N = len(colors)
        # diffs[i] = colors[i] XOR colors[(i+1)%N]
        diffs = [colors[i] ^ colors[(i+1)%N] for i in range(N)]
        
        # Find break points where diffs[i] == 0
        break_points = set(i for i in range(N) if diffs[i] == 0)
        
        # Function to compute run lengths based on break points
        def compute_run_lengths():
            run_lengths = []
            # Sort break points
            sorted_break_points = sorted(break_points)
            if not sorted_break_points:
                # No break points, one full circle run
                run_lengths.append(N)
            else:
                # Compute distances between consecutive break points, considering circularity
                for i in range(len(sorted_break_points)):
                    current = sorted_break_points[i]
                    next_bp = sorted_break_points[(i+1)%len(sorted_break_points)]
                    if next_bp > current:
                        run_length = next_bp - current - 1
                    else:
                        run_length = N - current - 1 + next_bp
                    run_lengths.append(run_length)
            return run_lengths
        
        # Compute initial run lengths
        run_lengths = compute_run_lengths()
        
        # Frequency count of run lengths
        freq = {}
        for l in run_lengths:
            freq[l] = freq.get(l, 0) + 1
        
        # Precompute prefix sums for frequency
        max_run = max(run_lengths) if run_lengths else 0
        prefix_freq = [0] * (max_run + 2)
        for l in range(max_run, -1, -1):
            prefix_freq[l] = freq.get(l, 0)
            if l + 1 <= max_run:
                prefix_freq[l] += prefix_freq[l + 1]
        
        # To store answers for type 1 queries
        answer = []
        
        for q in queries:
            if q[0] == 1:
                # Query type 1: count of alternating groups with size q[1]
                S = q[1]
                if S - 1 <= max_run:
                    answer.append(prefix_freq[S - 1])
                else:
                    answer.append(0)
            else:
                # Query type 2: update color at index q[1] to q[2]
                j = q[1]
                new_color = q[2]
                old_color = colors[j]
                if old_color != new_color:
                    # Update diffs[j-1] and diffs[j]
                    j_prev = (j - 1 + N) % N
                    diffs[j_prev] = colors[j_prev] ^ new_color
                    diffs[j] = new_color ^ colors[(j + 1) % N]
                    colors[j] = new_color
                    # Update break_points set
                    if diffs[j_prev] == 0:
                        break_points.add(j_prev)
                    else:
                        break_points.discard(j_prev)
                    if diffs[j] == 0:
                        break_points.add(j)
                    else:
                        break_points.discard(j)
                    # Recompute run_lengths and freq
                    run_lengths = compute_run_lengths()
                    # Clear freq and rebuild
                    freq = {}
                    for l in run_lengths:
                        freq[l] = freq.get(l, 0) + 1
                    # Update prefix_freq
                    max_run = max(run_lengths) if run_lengths else 0
                    prefix_freq = [0] * (max_run + 2)
                    for l in range(max_run, -1, -1):
                        prefix_freq[l] = freq.get(l, 0)
                        if l + 1 <= max_run:
                            prefix_freq[l] += prefix_freq[l + 1]
        
        return answer