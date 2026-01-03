

link : https://js.langchain.com/docs/how_to/installation/#installation-1


npm i dotenv



npm install @langchain/community @langchain/core

npm install langchain @langchain/core

npm i @langchain/cerebras @langchain/core

npm i zod






import { PromptTemplate, ChatPromptTemplate } from '@langchain/core/prompts'



const prompt = ChatPromptTemplate.fromMessages([
    [
        "system",
        ` You are a professional Math Expert, your job is to solve user questions.
        Think step by step through your reasonning and explain your thoughts.

        Instruction :
        - return only  the value of x 
        `,
    ],
    ["user", "here's the user question {input}"],
]);

const chain = prompt.pipe(llm)
const chainResult = await chain.invoke


// const prompt= PromptTemplate.fromTemplate(`
//         You are a professional Math Expert, your job is to solve user questions.
//         Think step by step through your reasonning and explain your thought

//         here's the user question
//         {input}

//         `)
// const invokePrompt=await prompt.invoke({input:"x+y=0, what is the value of x"})

// const result=await llm.invoke(invokePrompt)










COURSE STRUCTURE
- font [installation,login with google, design project page,dashboard,a a simple canvas]
- integrations[notion,drive, etc...]