#!/bin/bash
# Run individual test cases to collect compiler artifacts

# Set DTI_PROJECT_ROOT and activate virtual environment
export DTI_PROJECT_ROOT="$HOME/dt-inductor"
source $DTI_PROJECT_ROOT/torch-spyre-docs/scripts/dev-env.sh

# Test name passed as argument
TEST_NAME=$1

if [ -z "$TEST_NAME" ]; then
    echo "Usage: $0 <test_name>"
    echo "Example: $0 test_multiarg_pointwise_with_broadcast"
    exit 1
fi

# Create a unique work directory for this test
WORK_DIR=$(mktemp -d -t pytest_work_${TEST_NAME}_XXXXXX)
echo "Created work directory: $WORK_DIR"

# Enable debug mode to collect compiler artifacts
export TORCH_COMPILE_DEBUG="1"
export DXP_DEBUG="1"

# Set unique directories
export PYTEST_CACHE_DIR="$WORK_DIR/.pytest_cache"
export TORCH_COMPILE_DEBUG_DIR="$WORK_DIR/torch_compile_debug"
export TORCHINDUCTOR_CACHE_DIR="$WORK_DIR/torchinductor_cache"

# Print environment configuration
echo "Environment configuration:"
echo "  DTI_PROJECT_ROOT=$DTI_PROJECT_ROOT"
echo "  TORCH_COMPILE_DEBUG=$TORCH_COMPILE_DEBUG"
echo "  DXP_DEBUG=$DXP_DEBUG"
echo "  WORK_DIR=$WORK_DIR"
echo ""

# Run the specific test
cd torch-spyre/tests/inductor
python3 -c "
import sys
sys.path.insert(0, '.')
from test_phase3_multiarg import ${TEST_NAME}
try:
    ${TEST_NAME}()
    print('\\n✓ Test ${TEST_NAME} passed')
except Exception as e:
    print(f'\\n✗ Test ${TEST_NAME} failed: {e}')
    import traceback
    traceback.print_exc()
"

# Save the work directory path for later inspection
echo ""
echo "Compiler artifacts saved in: $WORK_DIR"
echo "To inspect artifacts:"
echo "  ls -la $WORK_DIR/torchinductor_cache/"
echo "  find $WORK_DIR -name '*.py' -o -name '*.txt' -o -name '*.json'"

# Keep the work directory (don't delete it)
echo ""
echo "Work directory preserved for inspection: $WORK_DIR"

# Made with Bob
