class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        initial_total = sum(nums1)
        total_increment = sum(nums2)
        max_time = n  # since we have at most n indices to reset

        N = n
        # For each index i, find the time s_i that maximizes the reduction
        reductions = []
        for i in range(n):
            s_i = N  # Start from the latest time
            max_reduction = nums1[i] + N * nums2[i]
            max_s = N
            for s in range(1, N):
                reduction = nums1[i] + s * nums2[i]
                if reduction >= max_reduction:
                    max_reduction = reduction
                    max_s = s
            reductions.append((i, max_s, max_reduction))

        # Sort the jobs in decreasing order of reduction
        reductions.sort(key=lambda x: -x[2])

        # Initialize time slots
        time_slots = [False] * (N + 1)  # index 0 unused, time slots from 1..N
        scheduled = []
        for i, s_i, _ in reductions:
            # Try to schedule at time s_i down to 1
            for s in range(s_i, 0, -1):
                if not time_slots[s]:
                    time_slots[s] = True
                    scheduled.append((s, i))
                    break

        # Now, try to find the minimal t such that effective sum <= x
        # We can consider t from 0 to N (max time slot scheduled)
        # For each t, compute the total reduction from scheduled jobs up to time t

        # Total reductions at each time t
        total_reduction = [0] * (N + 1)  # index 0 unused
        for s, i in scheduled:
            reduction = nums1[i] + s * nums2[i]
            total_reduction[s] = reduction

        # Compute cumulative reductions
        cumulative_reduction = [0] * (N + 1)  # index 0 unused
        for t in range(1, N + 1):
            cumulative_reduction[t] = cumulative_reduction[t - 1] + total_reduction[t]

        # Try t from 0 to N
        found = False
        for t in range(N + 1):
            effective_sum = initial_total + t * total_increment - cumulative_reduction[t]
            if effective_sum <= x:
                return t
        return -1