// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The LanceDB Authors

use std::sync::Mutex;

use lancedb::index::scalar::{BTreeIndexBuilder, FtsIndexBuilder};
use lancedb::index::vector::{
    IvfFlatIndexBuilder, IvfHnswPqIndexBuilder, IvfHnswSqIndexBuilder, IvfPqIndexBuilder,
};
use lancedb::index::Index as LanceDbIndex;
use napi_derive::napi;

use crate::util::parse_distance_type;

#[napi]
pub struct Index {
    inner: Mutex<Option<LanceDbIndex>>,
}

impl Index {
    pub fn consume(&self) -> napi::Result<LanceDbIndex> {
        self.inner
            .lock()
            .unwrap()
            .take()
            .ok_or(napi::Error::from_reason(
                "attempt to use an index more than once",
            ))
    }
}

#[napi]
impl Index {
    #[napi(factory)]
    pub fn ivf_pq(
        distance_type: Option<String>,
        num_partitions: Option<u32>,
        num_sub_vectors: Option<u32>,
        num_bits: Option<u32>,
        max_iterations: Option<u32>,
        sample_rate: Option<u32>,
    ) -> napi::Result<Self> {
        let mut ivf_pq_builder = IvfPqIndexBuilder::default();
        if let Some(distance_type) = distance_type {
            let distance_type = parse_distance_type(distance_type)?;
            ivf_pq_builder = ivf_pq_builder.distance_type(distance_type);
        }
        if let Some(num_partitions) = num_partitions {
            ivf_pq_builder = ivf_pq_builder.num_partitions(num_partitions);
        }
        if let Some(num_sub_vectors) = num_sub_vectors {
            ivf_pq_builder = ivf_pq_builder.num_sub_vectors(num_sub_vectors);
        }
        if let Some(num_bits) = num_bits {
            ivf_pq_builder = ivf_pq_builder.num_bits(num_bits);
        }
        if let Some(max_iterations) = max_iterations {
            ivf_pq_builder = ivf_pq_builder.max_iterations(max_iterations);
        }
        if let Some(sample_rate) = sample_rate {
            ivf_pq_builder = ivf_pq_builder.sample_rate(sample_rate);
        }
        Ok(Self {
            inner: Mutex::new(Some(LanceDbIndex::IvfPq(ivf_pq_builder))),
        })
    }

    #[napi(factory)]
    pub fn ivf_flat(
        distance_type: Option<String>,
        num_partitions: Option<u32>,
        max_iterations: Option<u32>,
        sample_rate: Option<u32>,
    ) -> napi::Result<Self> {
        let mut ivf_flat_builder = IvfFlatIndexBuilder::default();
        if let Some(distance_type) = distance_type {
            let distance_type = parse_distance_type(distance_type)?;
            ivf_flat_builder = ivf_flat_builder.distance_type(distance_type);
        }
        if let Some(num_partitions) = num_partitions {
            ivf_flat_builder = ivf_flat_builder.num_partitions(num_partitions);
        }
        if let Some(max_iterations) = max_iterations {
            ivf_flat_builder = ivf_flat_builder.max_iterations(max_iterations);
        }
        if let Some(sample_rate) = sample_rate {
            ivf_flat_builder = ivf_flat_builder.sample_rate(sample_rate);
        }
        Ok(Self {
            inner: Mutex::new(Some(LanceDbIndex::IvfFlat(ivf_flat_builder))),
        })
    }

    #[napi(factory)]
    pub fn btree() -> Self {
        Self {
            inner: Mutex::new(Some(LanceDbIndex::BTree(BTreeIndexBuilder::default()))),
        }
    }

    #[napi(factory)]
    pub fn bitmap() -> Self {
        Self {
            inner: Mutex::new(Some(LanceDbIndex::Bitmap(Default::default()))),
        }
    }

    #[napi(factory)]
    pub fn label_list() -> Self {
        Self {
            inner: Mutex::new(Some(LanceDbIndex::LabelList(Default::default()))),
        }
    }

    #[napi(factory)]
    #[allow(clippy::too_many_arguments)]
    pub fn fts(
        with_position: Option<bool>,
        base_tokenizer: Option<String>,
        language: Option<String>,
        max_token_length: Option<u32>,
        lower_case: Option<bool>,
        stem: Option<bool>,
        remove_stop_words: Option<bool>,
        ascii_folding: Option<bool>,
        ngram_min_length: Option<u32>,
        ngram_max_length: Option<u32>,
        prefix_only: Option<bool>,
    ) -> Self {
        let mut opts = FtsIndexBuilder::default();
        if let Some(with_position) = with_position {
            opts = opts.with_position(with_position);
        }
        if let Some(base_tokenizer) = base_tokenizer {
            opts = opts.base_tokenizer(base_tokenizer);
        }
        if let Some(language) = language {
            opts = opts.language(&language).unwrap();
        }
        if let Some(max_token_length) = max_token_length {
            opts = opts.max_token_length(Some(max_token_length as usize));
        }
        if let Some(lower_case) = lower_case {
            opts = opts.lower_case(lower_case);
        }
        if let Some(stem) = stem {
            opts = opts.stem(stem);
        }
        if let Some(remove_stop_words) = remove_stop_words {
            opts = opts.remove_stop_words(remove_stop_words);
        }
        if let Some(ascii_folding) = ascii_folding {
            opts = opts.ascii_folding(ascii_folding);
        }
        if let Some(ngram_min_length) = ngram_min_length {
            opts = opts.ngram_min_length(ngram_min_length);
        }
        if let Some(ngram_max_length) = ngram_max_length {
            opts = opts.ngram_max_length(ngram_max_length);
        }
        if let Some(prefix_only) = prefix_only {
            opts = opts.ngram_prefix_only(prefix_only);
        }

        Self {
            inner: Mutex::new(Some(LanceDbIndex::FTS(opts))),
        }
    }

    #[napi(factory)]
    pub fn hnsw_pq(
        distance_type: Option<String>,
        num_partitions: Option<u32>,
        num_sub_vectors: Option<u32>,
        max_iterations: Option<u32>,
        sample_rate: Option<u32>,
        m: Option<u32>,
        ef_construction: Option<u32>,
    ) -> napi::Result<Self> {
        let mut hnsw_pq_builder = IvfHnswPqIndexBuilder::default();
        if let Some(distance_type) = distance_type {
            let distance_type = parse_distance_type(distance_type)?;
            hnsw_pq_builder = hnsw_pq_builder.distance_type(distance_type);
        }
        if let Some(num_partitions) = num_partitions {
            hnsw_pq_builder = hnsw_pq_builder.num_partitions(num_partitions);
        }
        if let Some(num_sub_vectors) = num_sub_vectors {
            hnsw_pq_builder = hnsw_pq_builder.num_sub_vectors(num_sub_vectors);
        }
        if let Some(max_iterations) = max_iterations {
            hnsw_pq_builder = hnsw_pq_builder.max_iterations(max_iterations);
        }
        if let Some(sample_rate) = sample_rate {
            hnsw_pq_builder = hnsw_pq_builder.sample_rate(sample_rate);
        }
        if let Some(m) = m {
            hnsw_pq_builder = hnsw_pq_builder.num_edges(m);
        }
        if let Some(ef_construction) = ef_construction {
            hnsw_pq_builder = hnsw_pq_builder.ef_construction(ef_construction);
        }
        Ok(Self {
            inner: Mutex::new(Some(LanceDbIndex::IvfHnswPq(hnsw_pq_builder))),
        })
    }

    #[napi(factory)]
    pub fn hnsw_sq(
        distance_type: Option<String>,
        num_partitions: Option<u32>,
        max_iterations: Option<u32>,
        sample_rate: Option<u32>,
        m: Option<u32>,
        ef_construction: Option<u32>,
    ) -> napi::Result<Self> {
        let mut hnsw_sq_builder = IvfHnswSqIndexBuilder::default();
        if let Some(distance_type) = distance_type {
            let distance_type = parse_distance_type(distance_type)?;
            hnsw_sq_builder = hnsw_sq_builder.distance_type(distance_type);
        }
        if let Some(num_partitions) = num_partitions {
            hnsw_sq_builder = hnsw_sq_builder.num_partitions(num_partitions);
        }
        if let Some(max_iterations) = max_iterations {
            hnsw_sq_builder = hnsw_sq_builder.max_iterations(max_iterations);
        }
        if let Some(sample_rate) = sample_rate {
            hnsw_sq_builder = hnsw_sq_builder.sample_rate(sample_rate);
        }
        if let Some(m) = m {
            hnsw_sq_builder = hnsw_sq_builder.num_edges(m);
        }
        if let Some(ef_construction) = ef_construction {
            hnsw_sq_builder = hnsw_sq_builder.ef_construction(ef_construction);
        }
        Ok(Self {
            inner: Mutex::new(Some(LanceDbIndex::IvfHnswSq(hnsw_sq_builder))),
        })
    }
}
