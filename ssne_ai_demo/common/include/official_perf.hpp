#pragma once

void OfficialPerfReset(const char* feature_name, float sensor_fps = 90.0f);
void OfficialPerfFinishPreviousFrame();
void OfficialPerfBeginFrame();
void OfficialPerfPrintFinal();

