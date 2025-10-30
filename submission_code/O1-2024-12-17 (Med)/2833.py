class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        # ----------------------------------------------------------
        # We'll solve this using an "offline queries + Fenwick Tree" approach.
        #
        # Key idea:
        # - Sort all logs by their request time.
        # - Process queries in ascending order of query time.
        # - Maintain for each server the "latest request time" processed so far.
        # - Use a Fenwick Tree (Binary Indexed Tree) over possible times (0..10^6).
        #   Each index t in Fenwick Tree will store how many servers currently have "latest request time = t".
        #
        # For each server, initially latest request time = 0 (meaning it has no requests).
        # We'll keep them all counted at time=0 in Fenwick Tree at the beginning.
        #
        # Steps:
        # 1) Sort logs by time ascending.
        # 2) Sort queries by time ascending, but keep track of original index.
        # 3) For each query T in ascending order:
        #    - Process all logs whose time <= T:
        #         for each log (server, t), update the Fenwick Tree:
        #            remove the server from its old latest time
        #            add the server to t
        #         update that server's "latest time"
        #    - To find how many servers have a request in [T - x, T]:
        #         we count how many servers have "latest time" >= (T - x)
        #         i.e. do fenwicksum.range_sum(T-x, T).
        #      The answer to the query (servers with zero requests in that interval)
        #      is n - that_count.
        #
        # Because x < T, T - x >= 1, so we never need to clamp negative indices.
        #
        # Time complexity: O((len(logs) + len(queries)) * log(10^6)) which should be acceptable.
        # ----------------------------------------------------------

        # Fenwick Tree (Binary Indexed Tree) helpers
        # We'll store counts for indices in [0..max_time], using 1-based indexing internally.
        MAX_TIME = 10**6
        fenwicksz = MAX_TIME + 2  # +1 for 1-based, +1 for safety boundary
        fenw = [0] * fenwicksz

        def fenw_update(pos: int, delta: int) -> None:
            while pos < fenwicksz:
                fenw[pos] += delta
                pos += pos & -pos

        def fenw_prefix_sum(pos: int) -> int:
            s = 0
            while pos > 0:
                s += fenw[pos]
                pos -= pos & -pos
            return s

        def fenw_range_sum(left: int, right: int) -> int:
            # sum over [left, right], 1-based indexes
            if left > right:
                return 0
            return fenw_prefix_sum(right) - fenw_prefix_sum(left - 1)

        # convert logs' server_id to 0-based
        # (Note: problem statement server ids are in [1..n],
        #  here we'll shift them to [0..n-1].)
        for i in range(len(logs)):
            logs[i][0] -= 1

        # sort logs by time
        logs.sort(key=lambda x: x[1])

        # prepare Fenwick tree
        # initially all servers have latest_request_time = 0
        # so we put them (count n) in fenw[1].
        # We'll store times as: index = time + 1 for Fenwick ops
        fenw_update(1, n)

        # keep track of each server's last request time (0 means none so far)
        last_time = [0] * n

        # pair queries with their original indices: (query_time, idx)
        q_with_idx = [(queries[i], i) for i in range(len(queries))]
        q_with_idx.sort(key=lambda x: x[0])

        res = [0] * len(queries)

        log_ptr = 0
        L = len(logs)

        # process queries in ascending order
        for (T, qidx) in q_with_idx:
            # process all logs with time <= T
            while log_ptr < L and logs[log_ptr][1] <= T:
                s, t = logs[log_ptr]
                old_t = last_time[s]
                # remove from old time
                if old_t == 0:
                    fenw_update(1, -1)
                else:
                    fenw_update(old_t + 1, -1)
                # add to new time
                fenw_update(t + 1, 1)
                # update last_time for this server
                last_time[s] = t
                log_ptr += 1

            # count how many servers have last_time in [T-x, T]
            # i.e. last_time >= (T - x)
            # We'll do Fenwick range: [ (T - x) , T ]
            left = (T - x)  # we know T - x >= 1
            right = T
            # map to Fenwick: left+1 ... right+1
            active_count = fenw_range_sum(left + 1, right + 1)
            # number of servers with no requests in [T-x, T]
            res[qidx] = n - active_count

        return res