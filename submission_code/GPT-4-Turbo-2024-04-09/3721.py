class Solution:
    def countMentions(self, numberOfUsers: int, events: List[List[str]]) -> List[int]:
        mentions = [0] * numberOfUsers
        online_status = [True] * numberOfUsers
        offline_until = [-1] * numberOfUsers

        events.sort(key=lambda x: int(x[1]))  # Sort events by timestamp to process in order

        for event in events:
            event_type, timestamp_str, data = event
            timestamp = int(timestamp_str)

            # Update online status based on the current timestamp
            for user_id in range(numberOfUsers):
                if offline_until[user_id] != -1 and timestamp >= offline_until[user_id]:
                    online_status[user_id] = True
                    offline_until[user_id] = -1

            if event_type == "OFFLINE":
                user_id = int(data)
                online_status[user_id] = False
                offline_until[user_id] = timestamp + 60

            elif event_type == "MESSAGE":
                mentioned_ids = set()
                if data == "ALL":
                    mentioned_ids.update(range(numberOfUsers))
                elif data == "HERE":
                    mentioned_ids.update(i for i in range(numberOfUsers) if online_status[i])
                else:
                    tokens = data.split()
                    for token in tokens:
                        if token.startswith("id"):
                            id_num = int(token[2:])
                            mentioned_ids.add(id_num)

                for user_id in mentioned_ids:
                    mentions[user_id] += 1

        return mentions