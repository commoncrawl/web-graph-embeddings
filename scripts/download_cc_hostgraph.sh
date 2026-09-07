#!/bin/bash
# Download a Common Crawl HOST-level web graph (plain-text vertex + edge shards) for a release.
#
# Releases: https://commoncrawl.org/web-graphs  (default below is the project target).
# Each release publishes, under
#   https://data.commoncrawl.org/projects/hyperlinkgraph/<release>/host/
# gzipped *manifests* listing the actual shard files:
#   <release>-host-vertices.paths.gz   (lists the vertex part-*.txt.gz shards)
#   <release>-host-edges.paths.gz      (lists the edge   part-*.txt.gz shards)
# plus a WebGraph/BVGraph form (<release>-host.graph / -t.graph) and ranks.
#
# This script downloads the manifests and then every shard they reference into
#   <out>/vertices/  and  <out>/edges/
# which `wgl data prepare --vertices <out>/vertices --edges <out>/edges` ingests directly.
#
# Usage:
#   scripts/download_cc_hostgraph.sh                       # default release -> data/cc_host
#   scripts/download_cc_hostgraph.sh <release> <out-dir>
set -euo pipefail

RELEASE="${1:-cc-main-2026-mar-apr-may}"
OUT="${2:-data/cc_host}"
DATA_BASE="https://data.commoncrawl.org"
HOST_BASE="${DATA_BASE}/projects/hyperlinkgraph/${RELEASE}/host"

mkdir -p "${OUT}/vertices" "${OUT}/edges"
echo "Downloading ${RELEASE} host graph into ${OUT} ..."

# Fetch a manifest, then download each shard it lists (paths are relative to data.commoncrawl.org).
fetch_shards() {
    local kind="$1" subdir="$2"
    local manifest="${OUT}/${RELEASE}-host-${kind}.paths.gz"
    curl -fSL "${HOST_BASE}/${RELEASE}-host-${kind}.paths.gz" -o "${manifest}"
    echo "Manifest: ${manifest}"
    zcat "${manifest}" | while read -r path; do
        [ -z "${path}" ] && continue
        curl -fSL --create-dirs "${DATA_BASE}/${path}" -o "${OUT}/${subdir}/$(basename "${path}")"
    done
}

fetch_shards vertices vertices
fetch_shards edges edges

echo "Done. Prepare a 1% downsample with:"
echo "  wgl data prepare --vertices ${OUT}/vertices --edges ${OUT}/edges --out ${OUT} --keep-fraction 0.01"
echo "Full graph (~262M nodes / ~8.1B edges) needs the out-of-core / BVGraph path — see docs/runbook.md."
