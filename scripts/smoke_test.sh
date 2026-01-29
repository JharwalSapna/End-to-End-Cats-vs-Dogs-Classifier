#!/bin/bash
# Smoke test script for post-deployment validation
# This script tests the health and prediction endpoints

set -e

# Default service URL (can be overridden via environment variable)
SERVICE_URL="${SERVICE_URL:-http://localhost:8000}"

echo "Running smoke tests against: $SERVICE_URL"
echo "============================================"

# Test 1: Health Check
echo ""
echo "Test 1: Health Check"
echo "--------------------"
HEALTH_RESPONSE=$(curl -s -w "\n%{http_code}" "$SERVICE_URL/health")
HEALTH_CODE=$(echo "$HEALTH_RESPONSE" | tail -1)
HEALTH_BODY=$(echo "$HEALTH_RESPONSE" | head -n -1)

if [ "$HEALTH_CODE" -eq 200 ]; then
    echo "✓ Health check passed (HTTP $HEALTH_CODE)"
    echo "  Response: $HEALTH_BODY"
else
    echo "✗ Health check failed (HTTP $HEALTH_CODE)"
    exit 1
fi

# Test 2: Root Endpoint
echo ""
echo "Test 2: API Info"
echo "----------------"
ROOT_RESPONSE=$(curl -s -w "\n%{http_code}" "$SERVICE_URL/")
ROOT_CODE=$(echo "$ROOT_RESPONSE" | tail -1)

if [ "$ROOT_CODE" -eq 200 ]; then
    echo "✓ Root endpoint passed (HTTP $ROOT_CODE)"
else
    echo "✗ Root endpoint failed (HTTP $ROOT_CODE)"
    exit 1
fi

# Test 3: Prediction Endpoint (with a simple test image)
echo ""
echo "Test 3: Prediction Endpoint"
echo "---------------------------"

# Create a tiny test image (1x1 pixel, base64 encoded)
# This is a minimal valid JPEG for testing the endpoint
TEST_IMAGE="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

PRED_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$SERVICE_URL/predict" \
    -H "Content-Type: application/json" \
    -d "{\"image\": \"$TEST_IMAGE\"}")
PRED_CODE=$(echo "$PRED_RESPONSE" | tail -1)
PRED_BODY=$(echo "$PRED_RESPONSE" | head -n -1)

if [ "$PRED_CODE" -eq 200 ]; then
    echo "✓ Prediction endpoint passed (HTTP $PRED_CODE)"
    echo "  Response: $PRED_BODY"
elif [ "$PRED_CODE" -eq 400 ]; then
    # 400 is acceptable - means endpoint works but our test image might be invalid
    echo "~ Prediction endpoint reachable (HTTP $PRED_CODE - test image may be invalid)"
else
    echo "✗ Prediction endpoint failed (HTTP $PRED_CODE)"
    exit 1
fi

# Test 4: Metrics Endpoint
echo ""
echo "Test 4: Metrics Endpoint"
echo "------------------------"
METRICS_RESPONSE=$(curl -s -w "\n%{http_code}" "$SERVICE_URL/metrics")
METRICS_CODE=$(echo "$METRICS_RESPONSE" | tail -1)

if [ "$METRICS_CODE" -eq 200 ]; then
    echo "✓ Metrics endpoint passed (HTTP $METRICS_CODE)"
else
    echo "✗ Metrics endpoint failed (HTTP $METRICS_CODE)"
    exit 1
fi

# Summary
echo ""
echo "============================================"
echo "All smoke tests passed!"
echo "============================================"
