# src/envs/tool_free_env.py
import logging
from typing import Dict, Any, Optional, List

from .http_mcp_search_env import HttpMCPSearchEnv

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TOOL_FREE = """You are a Multi-Modal Reasoning Agent equipped with capabilities in multimodal analysis and reasoning.
Your task is to analyze the provided images and question, then generate a final answer supported by step-by-step reasoning.

### **ReAct Framework**

You must operate using a ReAct-style reasoning process. The reasoning must be explicit, structured, and visible in text.
- Think step-by-step about what information you can extract from the images.
- Analyze the visual content carefully.
- Combine visual information with your background knowledge.
- Mark key intermediate conclusions.
- Reach a final answer based on your reasoning.

You must not skip the reasoning process or jump to the final answer directly. Show your thinking clearly before concluding.

### **Overall Goal**

- Analyze visual information in the provided images;
- Perform cross-modal reasoning based on image content and background knowledge;
- Explicitly tag key intermediate reasoning results to support process-level evaluation;
- Provide concise, deterministic, and reviewable final answers.

------

### **Image Token Rules**

- Input images to the task:
  `<image_1> ... </image_1>`, `<image_2> ... </image_2>`, etc.
- When referencing images in your reasoning, use the correct image tokens.
- Carefully observe all visual details in the images provided.

------

### **Subgoal Tagging (for Process-Level Evaluation)**

**Tagging Principles**

Use `<SUBGOAL>` and `</SUBGOAL>` tags to wrap key intermediate reasoning conclusions or confirmed important information.

**`<SUBGOAL>` tags are intended to capture fact-level, verifiable intermediate conclusions, such as time, location, entities, events, relations, and quantities, that directly support the final answer. Sub-goals must not include guesses, common-sense assumptions, or reasoning shortcuts.**

**`<SUBGOAL>` tags should be generated dynamically and incrementally throughout the reasoning process.** As soon as an intermediate fact or validated insight is established—regardless of which reasoning step you are in—output the corresponding `<SUBGOAL>` tag immediately. **Do not wait until the final response to output all subgoals.**

**Strict Requirements**

- Each subgoal must be:
  an intermediate reasoning result or a fact confirmed through analysis;
- Must not be:
  operational steps, vague thoughts, or low-level details;
- Typically one sentence only, marking nodes of key information directly supporting the final answer;
- Avoid redundancy, repetition, or irrelevant tagging.
- Do NOT output <SUBGOAL> in the final turn unless it appears BEFORE the <FINAL_ANSWER>.

**Examples**
`<SUBGOAL>Confirm that the building in the image is the National Stadium (Bird's Nest) in Beijing.</SUBGOAL>`
`<SUBGOAL>Confirm that construction of the National Stadium in Beijing began on December 24, 2003.</SUBGOAL>`

------

### **Final Answer Format (Mandatory)**

- Wrap the final conclusion (maximum 1–2 sentences) strictly within:
  `<FINAL_ANSWER>Your final answer</FINAL_ANSWER>`
- The `<FINAL_ANSWER>` content must not contain reasoning steps, subgoals, or source citations.
- If you determine that the required final answer cannot be determined from the provided images and your knowledge, you must return:
`<FINAL_ANSWER>Failed. I cannot answer this question.</FINAL_ANSWER>`
- Do not fabricate or infer information beyond what can be reasonably concluded.
- Do NOT output multiple <FINAL_ANSWER> blocks.

------

### **Answering Rules**

1. Base your reasoning on the visual content in the images and your background knowledge;
2. When making claims, ensure they are grounded in observable image content or established facts;
3. Place reasoning process and subgoal tags outside the final answer;
4. Final answers must be concise and clear;
5. If the answer cannot be determined with reasonable confidence, explicitly declare failure using the required format.

"""


class ToolFreeEnv(HttpMCPSearchEnv):
    """
    Tool-Free environment: Single-turn conversation without tool access.
    
    This environment inherits from HttpMCPSearchEnv to reuse image handling
    and result extraction logic, but operates in a simplified mode:
    - No MCP server connection
    - No tool initialization
    - Single-turn conversation (one API call)
    - Direct reasoning based on images and question
    """

    has_heavy_resource = False

    def __init__(self, *args, **kwargs):
        # Disable gateway config to avoid MCP connection attempts
        kwargs["gateway_config_path"] = kwargs.get("gateway_config_path", "")
        
        # Initialize parent (will call _initialize_tools which we override)
        super().__init__(*args, **kwargs)
        
        # Ensure tools are cleared
        self.tool_schemas = []
        self.tool_descriptions = ""
        self.local_tools = {}
        self._tools_initialized = True
        
        logger.info(f"[{self.worker_id}] ToolFreeEnv initialized (no tools, single-turn mode)")

    @property
    def mode(self) -> str:
        return "tool_free"

    def _initialize_tools(self):
        """
        Override to disable tool initialization completely.
        No MCP server connection, no tool discovery.
        """
        self.tool_schemas = []
        self.tool_descriptions = ""
        self.local_tools = {}
        self._tools_initialized = True
        logger.debug(f"[{self.worker_id}] Tool initialization skipped (Tool-Free mode)")

    def _load_gateway_config(self, config_path: str) -> Dict[str, Any]:
        """
        Override to return empty config, avoiding resource dependencies.
        """
        return {"modules": []}

    def env_start(self):
        """
        No-op in Tool-Free mode - no MCP connection needed.
        """
        logger.info(f"[{self.worker_id}] ToolFreeEnv started (no MCP connection)")

    def get_system_prompt(
        self,
        task_question: Optional[str] = None,
        max_turns: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Return Tool-Free system prompt without tool descriptions.
        
        Key differences from parent:
        - No "Available Tools" section
        - No "Tool-Use Strategy" section
        - Emphasizes direct reasoning from images
        - Preserves SUBGOAL and FINAL_ANSWER formatting requirements
        """
        prompt = SYSTEM_PROMPT_TOOL_FREE
        
        # Add current task
        if task_question:
            prompt += f"\n\n## Current Task\n{task_question}"
        
        # Note: Turn budget not strictly needed for single-turn, but kept for consistency
        if isinstance(max_turns, int) and max_turns > 0:
            prompt += (
                "\n\n## Response Guidelines\n"
                "- Provide your complete reasoning and answer in a single response.\n"
                "- Use <SUBGOAL> tags for key intermediate conclusions.\n"
                "- End with <FINAL_ANSWER>...</FINAL_ANSWER>.\n"
            )
        
        return prompt

    def _run_conversation(
        self,
        question: str,
        model_name: str,
        max_turns: int,
        max_retries: int,
        logger: logging.Logger,
        task_timeout: Optional[float] = None,
        task_start_time: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Single-turn conversation: Send question + images, receive one response.
        
        Flow:
        1. Build system prompt (no tools)
        2. Build user message with question
        3. Inject images using parent's _append_input_images
        4. Call OpenAI API once (no tools parameter)
        5. Return messages list
        
        Args:
            question: Task question
            model_name: Model to use
            max_turns: Ignored in single-turn mode
            max_retries: Retry count for API failures
            logger: Logger instance
            task_timeout: Task timeout (inherited from parent)
            task_start_time: Task start time (inherited from parent)
            
        Returns:
            List of messages: [system, user, assistant]
        """
        from openai.types.chat import ChatCompletionMessageParam
        
        # Build system message
        system_prompt = self.get_system_prompt(question, max_turns=max_turns)
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
        ]
        
        # Build user message content
        user_content: List[Dict[str, Any]] = [
            {"type": "text", "text": f"Question: {question}\n"}
        ]
        
        # Inject input images using parent's method (handles <image_k> tagging)
        # This reuses the logic from HttpMCPSearchEnv._append_input_images
        self._append_input_images(user_content)
        
        messages.append({"role": "user", "content": user_content})
        
        # Get OpenAI client
        client = self._get_openai_client()
        
        # Single API call with retry logic
        retry = 0
        while retry < max_retries:
            try:
                # Call without tools parameter (key difference from parent)
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages
                )
                
                if not hasattr(response, "choices") or not response.choices:
                    raise ValueError("OpenAI API returned empty response")
                
                # Append assistant response
                assistant_message = response.choices[0].message
                messages.append(assistant_message.model_dump())
                
                logger.info(f"[{self.worker_id}] Single-turn conversation completed")
                return messages
                
            except Exception as exc:
                retry += 1
                logger.warning(f"[{self.worker_id}] API call failed (retry {retry}/{max_retries}): {exc}")
                if retry >= max_retries:
                    # Preserve partial messages for debugging
                    raise
        
        return messages
