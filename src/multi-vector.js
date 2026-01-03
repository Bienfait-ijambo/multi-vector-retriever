import { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { CohereEmbeddings } from "@langchain/cohere";
import { PineconeStore } from "@langchain/pinecone";
import { Pinecone as PineconeClient } from "@pinecone-database/pinecone";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { v4 as uuidv4 } from 'uuid';

import {
    ContextualCompressionRetriever
} from "@langchain/classic/retrievers/contextual_compression";
import { LLMChainExtractor } from "@langchain/classic/retrievers/document_compressors/chain_extract";

import { ChatCerebras } from "@langchain/cerebras";


import "dotenv/config";


// arrays
async function loadRawDocs(urls) {
    const allDocs = await Promise.all(
        urls.map(async (url) => {
            const loader = new CheerioWebBaseLoader(url);
            const docs = await loader.load();
            return docs.map(doc => {
                doc.metadata.originalUrl = url;  
                doc.metadata.source = url;       
                return doc;
            });
        })
    );
    return allDocs.flat();
}
// rawDocs is array
async function createParentDocs(rawDocs) {
    const parentSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });
    const parentSplits = await parentSplitter.splitDocuments(rawDocs);

    return parentSplits.map((split) => {
        const chunkId = uuidv4();  // UNIQUE ID per chunk
        split.metadata.docType = "parent";
        split.metadata.chunkId = chunkId;
        split.metadata.parentId = chunkId;  // Self-reference
        split.metadata.source = chunkId;
        return split;
    });
}

async function createChildDocs(parentDocs) {
    const childSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 400, chunkOverlap: 50 });
    const childSplits = await childSplitter.splitDocuments(parentDocs);

    return childSplits.map((split, i) => {
        // Get parent metadata for this child
        const parentIndex = Math.floor(i / 4); // ~4 children per parent chunk
        const parentMetadata = parentDocs[parentIndex]?.metadata;

        split.metadata.docType = "child";
        split.metadata.parentId = parentMetadata?.chunkId;  // Link to parent UUID
        split.metadata.chunkId = `child-${parentMetadata?.chunkId}-${i}`;
        split.metadata.source = split.metadata.chunkId;
        return split;
    });
}

export async function docEmbeddingMultiVector(urls) {
    const embeddings = new CohereEmbeddings({
        model: "embed-english-v3.0",
        apiKey: process.env.COHERE_API_KEY,
    });

    const pinecone = new PineconeClient({ apiKey: process.env.PINECONE_API_KEY });
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX);

    console.log(" Loading raw documents...");
    const rawDocs = await loadRawDocs(urls);

    console.log("Creating parent chunks...");
    const parentDocs = await createParentDocs(rawDocs);
    console.log('parentDoc   : ', parentDocs)

    console.log(" Creating child chunks...");
    const childDocs = await createChildDocs(parentDocs);

    console.log(" Storing in Pinecone...");
    const vectorStore = new PineconeStore(embeddings, {
        pineconeIndex,
        maxConcurrency: 5
    });

    // Store BOTH parent and child chunks in same index
    await vectorStore.addDocuments([...parentDocs, ...childDocs]);

    console.log(`✅ Single index: ${parentDocs.length} parent chunks + ${childDocs.length} child chunks`);
    console.log(`📊 Total documents: ${parentDocs.length + childDocs.length}`);

    console.log('finished embedding...')
}




export async function queryMultiVector(query, kParents = 3) {

    const embeddings = new CohereEmbeddings({
        model: "embed-english-v3.0",
        apiKey: process.env.COHERE_API_KEY,
    });

    const pinecone = new PineconeClient({ apiKey: process.env.PINECONE_API_KEY });
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX);

    const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
        pineconeIndex,
        maxConcurrency: 5,
    });


    // STEP 1: Child docs
    const childDocs = await vectorStore.similaritySearch(query, 10,
        { docType: "child" }
    );

    // STEP 2: FIXED parent matching using parentChunkId
    const parentChunkIds = [...new Set(childDocs.map(c => c.metadata.parentId))];

    // const parentDocs = await vectorStore.similaritySearch(query, kParents,
    //     {
    //         docType: "parent",
    //         source: { $in: parentChunkIds }  // Now matches exactly!
    //     }
    // );





    const compressor = LLMChainExtractor.fromLLM(
        new ChatCerebras({
            model: "llama-3.3-70b",
            temperature: 0,
            apiKey: process.env.CEREBRAS_API_KEY,
        })
    )


    const retriever = new ContextualCompressionRetriever({
        baseCompressor: compressor,
        baseRetriever: vectorStore.asRetriever({
            k: kParents,
            filter: {
                docType: "parent",
                source: { $in: parentChunkIds }
            }
        })


    })

    const retrievedDocs = await retriever.invoke(query)


    return {
        query,
        retrievedDocs
        // childMatches: childDocs.length,
        // parentChunkIds,
        // parentDocs,
    };
}

// await docEmbeddingMultiVector(['https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/']);


const results = await queryMultiVector("Types of prompt engineering");
// // // const results = await queryMultiVector("List all prompt engineering methods like zero-shot prompting, few-shot learning, chain-of-thought CoT, self-consistency, Tree of Thoughts, and instruction tuning?");
console.log('results   : ', results)
