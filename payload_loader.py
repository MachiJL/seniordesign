def load_payloads(filepath):
    payloads = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("Category"):
                continue

            payloads.append(line)

    return payloads