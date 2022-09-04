import sys
import os
import shutil
import time

TESTCASE_DIR = "/autograder/source/testcases/"
SUBMISSION_DIR = "/autograder/submission/"
RESULTS_FILE = "/autograder/results/results.json"
# TESTCASE_DIR = "testcases/"                 # for testing only
# SUBMISSION_DIR = "submission/"              # for testing only
# RESULTS_FILE = "scratch/results.json"       # for testing only


def main(arglist):
    t0 = time.time()

    # output version information
    import sys
    print("Python version")
    print(sys.version)
    print("Version info.")
    print(sys.version_info)

    # copy submission files to auto-grader working directory
    submission_files = os.listdir(SUBMISSION_DIR)
    for f in submission_files:
        if not os.path.isdir(SUBMISSION_DIR + f) and f != 'tester.py':
            shutil.copyfile(SUBMISSION_DIR + f, f)
            print(f"Copied {f}.")
        else:
            print("Omitting tester.")

    # import tester after student submission has been copied to the working directory
    from tester import main as run_tester

    # run tester on both methods + all testcases
    run_tester(['both', '1,2,3,4,5', '-l', RESULTS_FILE])
    # run_tester(['both', '1,2,3', '-l', RESULTS_FILE])     # for testing only

    # remove submission files from working directory
    for f in submission_files:
        os.remove(f)

    print(f'Autograder completed in {time.time() - t0} seconds.')


if __name__ == "__main__":
    main(sys.argv[1:])

