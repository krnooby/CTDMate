import argparse
from agent import run_agent

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CTD ReAct agent (planner + external tools)")
    p.add_argument("--files","-f", nargs="*", help="input files")
    p.add_argument("--texts","-t", nargs="*", help="inline texts")
    args = p.parse_args()

    out = run_agent(file_paths=args.files or [], texts=args.texts or [])
    print(out.get("final_message",""))
    if out.get("saved_path"):
        print("Saved:", out["saved_path"])
